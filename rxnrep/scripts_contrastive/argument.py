"""
Arguments for different parts of the model: encoder, mlp_diff, decoder...
"""


def dataset_args(parser, dataset: str):

    if "schneider" in dataset:
        prefix = "/Users/mjwen/Documents/Dataset/uspto/Schneider50k/"
        fname_tr = prefix + "schneider50k_n400_processed_train.tsv"
        fname_val = fname_tr
        fname_test = fname_tr

        # to require labels for classification
        if "classification" in dataset:
            parser.add_argument("--has_class_label", type=int, default=1)

    elif dataset == "electrolyte":
        prefix = "/Users/mjwen/Documents/Dataset/electrolyte/"
        fname_tr = prefix + "reactions_n2000_train.json"
        fname_val = prefix + "reactions_n2000_val.json"
        fname_test = prefix + "reactions_n2000_test.json"

        parser.add_argument(
            "--only_break_bond",
            type=int,
            default=1,
            help="whether the dataset has only breaking bond, i.e. no added bond",
        )

    elif dataset == "green":
        prefix = "/Users/mjwen/Documents/Dataset/activation_energy_Green/"
        fname_tr = prefix + "wb97xd3_n200_processed_train.tsv"
        fname_val = fname_tr
        fname_test = fname_tr

    else:
        raise ValueError(f"Not supported dataset {dataset}")

    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument("--trainset_filename", type=str, default=fname_tr)
    parser.add_argument("--valset_filename", type=str, default=fname_val)
    parser.add_argument("--testset_filename", type=str, default=fname_test)
    parser.add_argument(
        "--dataset_state_dict_filename", type=str, default="dataset_state_dict.yaml"
    )

    parser.add_argument("--allow_label_scaler_none", type=int, default=0)

    return parser


def training_args(parser):

    # restore
    parser.add_argument("--restore", type=int, default=0, help="restore training")

    # accelerator
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, default=None, help="number of gpus per node"
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="backend, e.g. `ddp`"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="number of processes for constructing graphs in dataset",
    )

    # training algorithm
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

    # learning rate
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        # default="reduce_on_plateau",
        default="cosine",
        help="`reduce_on_plateau` or cosine",
    )
    parser.add_argument("--lr_warmup_step", type=int, default=10)

    # this is only for cosine scheduler
    parser.add_argument("--lr_min", type=float, default=1e-6, help="min learning rate")

    return parser


def general_args(parser):

    # activation for all
    parser.add_argument("--activation", type=str, default="ReLU")

    return parser


def encoder_args(parser):

    parser.add_argument(
        "--conv",
        type=str,
        default="GatedGCNConv",
        # default="GINConvGlobal",
        choices=["GatedGCNConv", "GINConvGlobal", "GINConv"],
    )
    parser.add_argument("--has_global_feats", type=int, default=None)

    # embedding
    parser.add_argument("--embedding_size", type=int, default=None)

    # encoder
    parser.add_argument(
        "--molecule_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule_num_fc_layers", type=int, default=2)
    parser.add_argument("--molecule_batch_norm", type=int, default=1)
    parser.add_argument("--molecule_residual", type=int, default=1)
    parser.add_argument("--molecule_dropout", type=float, default="0.0")

    parser.add_argument("--reaction_conv_layer_sizes", type=int, nargs="+", default=[])
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # mlp diff
    parser.add_argument(
        "--mlp_diff_layer_sizes",
        type=int,
        nargs="+",
        default=None,
        help="`None` to not use it",
    )
    parser.add_argument("--mlp_diff_layer_batch_norm", type=int, default=1)

    # pool
    parser.add_argument(
        "--pool_method",
        type=str,
        default="set2set",
        help="set2set, hop_distance, global_only, sum_cat_all, sum_cat_center",
    )
    parser.add_argument("--pool_kwargs", type=str, default=None)

    # mlp pool
    parser.add_argument(
        "--mlp_pool_layer_sizes",
        type=int,
        nargs="+",
        default=None,
        help="`None` to not use it",
    )
    parser.add_argument("--mlp_pool_layer_batch_norm", type=int, default=1)

    return parser


def encoder_helper(parser):
    parser.add_argument(
        "--conv_layer_size",
        type=int,
        default=64,
        help="hidden layer size for mol and rxn conv",
    )
    parser.add_argument("--num_mol_conv_layers", type=int, default=2)
    parser.add_argument("--num_rxn_conv_layers", type=int, default=0)
    parser.add_argument("--num_mlp_diff_layers", type=int, default=0)
    parser.add_argument("--num_mlp_pool_layers", type=int, default=0)

    return parser


def encoder_adjuster(args):
    # conv
    if args.has_global_feats is None:
        if args.conv == "GINConv":
            args.has_global_feats = 0
        elif args.conv in ["GINConvGlobal", "GatedGCNConv"]:
            args.has_global_feats = 1
        else:
            raise ValueError("Unsupported conv")

    # embedding
    if args.embedding_size is None:
        args.embedding_size = args.conv_layer_size

    # encoder
    args.molecule_conv_layer_sizes = [args.conv_layer_size] * args.num_mol_conv_layers
    args.reaction_conv_layer_sizes = [args.conv_layer_size] * args.num_rxn_conv_layers

    args.mlp_diff_layer_sizes = [args.conv_layer_size] * args.num_mlp_diff_layers

    # mlp pool
    val = determine_layer_size_by_pool_method(args)
    args.mlp_pool_layer_size = [val] * args.num_mlp_pool_layers

    return args


def reaction_energy_decoder_args(parser):
    parser.add_argument(
        "--reaction_energy_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[64],
    )

    return parser


def activation_energy_decoder_args(parser):
    parser.add_argument(
        "--activation_energy_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[64],
    )

    return parser


def energy_decoder_helper(parser):
    parser.add_argument("--num_energy_decoder_layers", type=int, default=2)
    return parser


def reaction_energy_decoder_adjuster(args):
    val = determine_layer_size_by_pool_method(args)

    args.reaction_energy_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_energy_decoder_layers)
    ]
    return args


def activation_energy_decoder_adjuster(args):
    val = determine_layer_size_by_pool_method(args)

    args.activation_energy_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_energy_decoder_layers)
    ]
    return args


def reaction_type_decoder_args(parser):
    parser.add_argument(
        "--reaction_type_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[256, 128],
    )
    parser.add_argument("--num_reaction_classes", type=int, default=46)

    return parser


def reaction_type_decoder_helper(parser):
    parser.add_argument("--reaction_type_decoder_num_layers", type=int, default=2)
    return parser


def reaction_type_decoder_adjuster(args):
    val = determine_layer_size_by_pool_method(args)

    args.reaction_type_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.reaction_type_decoder_num_layers)
    ]

    return args


def simclr_decoder_args(parser):
    parser.add_argument(
        "--simclr_hidden_layer_sizes", type=int, nargs="+", default=[256, 128]
    )
    parser.add_argument("--simclr_temperature", type=float, default=0.1)

    return parser


def simclr_decoder_helper(parser):
    parser.add_argument("--simclr_num_layers", type=int, default=2)
    return parser


def simclr_decoder_adjuster(args):
    val = determine_layer_size_by_pool_method(args)
    minimum = args.conv_layer_size

    args.simclr_hidden_layer_sizes = [
        max(val // 2 ** i, minimum) for i in range(args.simclr_num_layers)
    ]

    return args


def finetune_args(parser):
    parser.add_argument(
        "--pretrained_dataset_state_dict_filename",
        type=str,
        default="pretrained_model/dataset_state_dict.yaml",
    )
    parser.add_argument(
        "--pretrained_config_filename", type=str, default="pretrained_model/config.yaml"
    )
    parser.add_argument(
        "--pretrained_ckpt_path", type=str, default="pretrained_model/checkpoint.ckpt"
    )
    parser.add_argument(
        "--pretrained_tune_encoder",
        type=int,
        default=0,
        help="Whether to optimize params in the encoder of the pretrained model. "
        "Note, parameters in the decoders are set to be fixed (since they are not used).",
    )

    return parser


def data_augmentation_args(parser):
    parser.add_argument(
        "--augment_1",
        type=str,
        default="drop_atom",
        choices=[
            "drop_atom",
            "drop_bond",
            "mask_atom",
            "mask_bond",
            "subgraph",
            "identity",
        ],
    )
    parser.add_argument(
        "--augment_2",
        type=str,
        default="drop_bond",
        choices=[
            "drop_atom",
            "drop_bond",
            "mask_atom",
            "mask_bond",
            "subgraph",
            "identity",
        ],
    )
    parser.add_argument("--augment_1_ratio", type=float, default=0.2)
    parser.add_argument("--augment_2_ratio", type=float, default=0.2)
    parser.add_argument("--augment_mask_value_atom", type=float, default=1.0)
    parser.add_argument("--augment_mask_value_bond", type=float, default=1.0)

    return parser


def determine_layer_size_by_pool_method(args):
    val = args.conv_layer_size

    if args.pool_method in ["set2set", "sum_cat_all", "sum_cat_center"]:
        val = val * 2
    elif args.pool_method == "global_only":
        val = val
    else:
        raise NotImplementedError

    return val
