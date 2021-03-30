"""
Arguments for different parts of the model: encoder, mlp_diff, decoder...
"""


def encoder_args(parser):

    # embedding
    parser.add_argument("--embedding_size", type=int, default=24)

    # encoder
    parser.add_argument(
        "--molecule_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule_num_fc_layers", type=int, default=2)
    parser.add_argument("--molecule_batch_norm", type=int, default=1)
    parser.add_argument("--molecule_activation", type=str, default="ReLU")
    parser.add_argument("--molecule_residual", type=int, default=1)
    parser.add_argument("--molecule_dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_activation", type=str, default="ReLU")
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # mlp_diff
    parser.add_argument(
        "--mlp_diff_layer_sizes",
        type=int,
        nargs="+",
        default=None,
        help="`None` to not use it",
    )
    parser.add_argument("--mlp_diff_layer_activation", type=str, default="ReLU")

    # pool
    parser.add_argument(
        "--pool_method",
        type=str,
        default="set2set",
        help="set2set, hop_distance, or global_only",
    )

    parser.add_argument(
        "--hop_distance_pool_max_hop_distance",
        type=int,
        default=2,
        help=(
            "max hop distance when hop_distance pool method is used. Ignored when "
            "`set2set` pool method is used. This is different from max_hop_distance "
            "used for node decoder, which is used to create labels for the decoders. "
            "Also, typically we can set the two to be the same."
        ),
    )

    return parser


def encoder_helper(parser):
    parser.add_argument(
        "--conv_layer_size",
        type=int,
        default=64,
        help="hidden layer size for mol and rxn conv",
    )
    parser.add_argument("--num_mol_conv_layers", type=int, default=2)
    parser.add_argument("--num_rxn_conv_layers", type=int, default=2)

    return parser


def encoder_adjuster(args):
    # encoder
    args.molecule_conv_layer_sizes = [args.conv_layer_size] * args.num_mol_conv_layers
    args.reaction_conv_layer_sizes = [args.conv_layer_size] * args.num_rxn_conv_layers

    if args.num_rxn_conv_layers == 0:
        args.reaction_dropout = 0

    # pool
    if args.pool_method in ["set2set", "global_only"]:
        args.pool_kwargs = None
    elif args.pool_method == "hop_distance":
        args.pool_kwargs = {"max_hop_distance": args.hop_distance_pool_max_hop_distance}
    else:
        raise NotImplementedError

    return args


def atom_bond_decoder_args(parser):

    # atom and bond decoder
    parser.add_argument(
        "--node_decoder_hidden_layer_sizes", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--node_decoder_activation", type=str, default="ReLU")
    parser.add_argument("--max_hop_distance", type=int, default=3)
    parser.add_argument("--atom_type_masker_ratio", type=float, default=0.2)
    parser.add_argument(
        "--atom_type_masker_use_masker_value",
        type=int,
        default=1,
        help="whether to use atom type masker value",
    )

    return parser


def atom_bond_decoder_helper(parser):
    parser.add_argument("--num_node_decoder_layers", type=int, default=1)
    return parser


def atom_bond_decoder_adjuster(args):
    val = get_encoder_out_feats_size(args)
    args.node_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_node_decoder_layers)
    ]

    return args


def kmeans_cluster_decoder_args(parser):
    parser.add_argument(
        "--num_centroids",
        type=int,
        nargs="+",
        default=[10],
        help="number of centroids for each clustering prototype",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature in the loss for cluster decoder",
    )
    parser.add_argument(
        "--num_kmeans_iterations",
        type=int,
        default=10,
        help="number of kmeans clustering iterations",
    )
    parser.add_argument(
        "--kmeans_similarity",
        type=str,
        default="cosine",
        help="similarity measure for kmeans: `cosine` or `euclidean`",
    )

    return parser


def kmeans_cluster_decoder_helper(parser):
    parser.add_argument("--prototype_size", type=int, default=10)
    parser.add_argument("--num_prototypes", type=int, default=1)

    return parser


def kmeans_cluster_decoder_adjuster(args):
    args.num_centroids = [args.prototype_size] * args.num_prototypes
    return args


def reaction_energy_decoder_args(parser):
    parser.add_argument(
        "--reaction_energy_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[64],
    )
    parser.add_argument(
        "--reaction_energy_decoder_activation", type=str, default="ReLU"
    )

    return parser


def activation_energy_decoder_args(parser):
    parser.add_argument(
        "--activation_energy_decoder_hidden_layer_sizes",
        type=int,
        nargs="+",
        default=[64],
    )

    parser.add_argument(
        "--activation_energy_decoder_activation", type=str, default="ReLU"
    )

    return parser


def energy_decoder_helper(parser):
    parser.add_argument("--num_energy_decoder_layers", type=int, default=2)
    return parser


def reaction_energy_decoder_adjuster(args):
    val = get_encoder_out_feats_size(args)

    if args.pool_method == "global_only":
        val = val
    else:
        val = 2 * val

    args.reaction_energy_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_energy_decoder_layers)
    ]
    return args


def activation_energy_decoder_adjuster(args):
    val = get_encoder_out_feats_size(args)

    if args.pool_method == "global_only":
        val = val
    else:
        val = 2 * val

    args.activation_energy_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.num_energy_decoder_layers)
    ]
    return args


def bep_label_args(parser):
    parser.add_argument(
        "--have_activation_energy_ratio",
        type=float,
        default=0.2,
        help=(
            "the ratio to use the activation energy, i.e. 1-ratio activation energies "
            "will be treated as unavailable."
        ),
    )
    parser.add_argument("--min_num_data_points_for_fitting", type=int, default=3)

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


def get_encoder_out_feats_size(args):
    """
    output atom/bond/global feature size, before pool
    """
    if args.mlp_diff_layer_sizes:
        encoder_out_feats_size = args.mlp_diff_layer_sizes[-1]
    else:
        encoder_out_feats_size = args.conv_layer_size

    return encoder_out_feats_size
