import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

import hessian_spectrum 

import csv 
from datetime import datetime

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, scheduler):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.training.learning_rate,
        total_steps=args.training.train_steps,
        pct_start=0.5,
        anneal_strategy='linear'
    )
    # set to 16 points throughout training 
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        noise_variance=0.25,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    def plot_hessian(model, train_data, ckpt_iteration):
        gradient_accumulation_steps = 60 # from original code
        use_minibatch = True
        # assumes gpu 
        device = 'cuda'
        ctx = torch.cuda.device(device)
        context_length = 1024 # gpt 2
        all = []
        last_layers = []
        for name, param in model.named_parameters():
            if '_backbone' in name:
                all.append(name)
            if '_backbone.h.11' in name or '_backbone.ln_f' in name or '_read_out' in name:
                last_layers.append(name)
        print(last_layers)
        hessian = hessian_spectrum.Hessian(model, 
                                           ckpt_iteration = ckpt_iteration, 
                                           train_data = train_data, 
                                           batch_size = args.training.batch_size, 
                                           block_size = context_length,  
                                           ctx = ctx, 
                                           use_minibatch = use_minibatch, 
                                           gradient_accumulation_steps = gradient_accumulation_steps, 
                                           device = device, 
                                           sample_layer = last_layers,
                                           comment = f"gpt2-4layer-icl-diversity-last-layers-lbl-{args.training.num_tasks}-tasks")

        hessian.get_spectrum(layer_by_layer = True)
        hessian.load_curve(layer_by_layer = True)

        hessian.get_spectrum(layer_by_layer = False)
        hessian.load_curve(layer_by_layer = False)

    last_xs = None
    last_ys = None
    last_loss = None

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        # Log hessian every 1000 steps 
        train_data = (xs, ys)
        # if i % 5000 == 0: 
           # plot_hessian(model, train_data, i)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, lr_scheduler)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        # Due to storage constraints 
        # if i % args.training.save_every_steps == 0 and not args.test_run:
            # training_state = {
                # "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                # "train_step": i,
            # }
            # torch.save(training_state, state_path)

        # if (
            # args.training.keep_every_steps > 0
            #  and i % args.training.keep_every_steps == 0
            # and i % 200000 == 0
            # and not args.test_run
            # and i > 0
        # ):
            # torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

        if i == len(pbar) - 1:
            last_xs = xs
            last_ys = ys
            last_loss = loss
    
    train_data = (last_xs, last_ys)
    
    # evaluate on T_True set
    t_true_task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        noise_variance=0.25,
        **args.training.task_kwargs,
    )
    ttrue_xs = data_sampler.sample_xs(
        curriculum.n_points,
        bsize,
        curriculum.n_dims_truncated,
        **data_sampler_args,
    )
    ttrue_task = t_true_task_sampler(**task_sampler_args)
    ttrue_ys = ttrue_task.evaluate(ttrue_xs)
    loss_func = t_true_task_sampler.get_training_metric()
    ttrue_output = model(ttrue_xs, ttrue_ys)
    ttrue_loss = loss_func(ttrue_output, ttrue_ys)
    print(f"ttrue_loss: {ttrue_loss}")
    # log the loss on T_Pretrain set and T_True set 
    metrics_file = os.path.join(args.out_dir, f'diversity_loss_metrics.csv')
    # Create file with headers if it doesn't exist
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['num_tasks', 't_pretrain_loss', 't_true_loss'])
    # Append new row
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            args.training.num_tasks,
            last_loss,
            ttrue_loss 
        ])
    # log hessian after training 
    plot_hessian(model, train_data, len(pbar) - 1)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_500000_{args.training.num_tasks}_tasks.pt"))
    


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
