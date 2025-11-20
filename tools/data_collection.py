from isaaclab_tasks.utils.data_collector import RobomimicDataCollector


def load_dataset_saver(args_cli, log_dir, iter):

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=f"{args_cli.filename}_{iter}",
        num_demos=1000,
        flush_freq=args_cli.num_envs,
        env_config={"device": args_cli.device},
    )
    collector_interface.reset()
    return collector_interface
