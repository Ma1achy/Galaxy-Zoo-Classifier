from packages.imports import argparse, datetime, os, tf
from dataframe.dataframe import DataFrame, get_paths
from network.network import Network
from programstate import global_state_dict, read_json

def main(
    seed,
    time_stamp,
    global_epoch,
    global_batch,
    saved_weights_path,
    save_step,
    root,
    dataset_root,
    train_catalog,
    test_catalog,
    checkpoint_root,
    log_root,
    classes,
    weight_mode,
    beta,
    build_paths,
    binarised_labels,
    ds_fraction,
    val_fraction,
    input_shape,
    input_dim,
    summary,
    architecture,
    learning_rate,
    loss_function,
    activation_function,
    weight_reg,
    num_patches,      
    projection_dim,   
    num_heads,        
    transformer_layers,
    mlp_head_units,   
    train_batch_size,
    val_batch_size,
    val_steps,
    epochs,
    steps_per_epoch,
    augmentation,
    crop,
    crop_size
    ):
    
    # global parameters
    global_state_dict["seed"] = seed
    global_state_dict["time_stamp"] = time_stamp
    global_state_dict["global_epoch"] = global_epoch
    global_state_dict["global_batch"] = global_batch
    
    # model saving parameters
    global_state_dict["saved_weights_path"] = saved_weights_path
    global_state_dict["save_step"] = save_step
    
    # file paths
    global_state_dict["root"] = root
    global_state_dict["dataset_root"] = dataset_root
    global_state_dict["train_catalog"] = train_catalog
    global_state_dict["test_catalog"] = test_catalog
    global_state_dict["checkpoint_root"] = checkpoint_root
    global_state_dict["log_root"] = log_root
    
    # dataset parameters
    global_state_dict["classes"] = classes
    global_state_dict["weight_mode"] = weight_mode
    global_state_dict["beta"] = beta
    global_state_dict["build_paths"] = build_paths
    global_state_dict["binarised_labels"] = binarised_labels
    global_state_dict["ds_fraction"] = ds_fraction
    global_state_dict["val_fraction"] = val_fraction
    
    # image properties
    global_state_dict["input_shape"] = input_shape
    global_state_dict["input_dim"] = input_dim
    
    # network parameters
    global_state_dict["summary"] = summary
    global_state_dict["architecture"] = architecture
    global_state_dict["learning_rate"] = learning_rate
    global_state_dict["loss_function"] = loss_function
    global_state_dict["activation_function"] = activation_function
    global_state_dict["weight_reg"] = weight_reg
    
    # transformer model hyperparameters
    
    global_state_dict["num_patches"] = num_patches
    global_state_dict["projection_dim"] = projection_dim
    global_state_dict["num_heads"] = num_heads
    global_state_dict["transformer_layers"] = transformer_layers
    global_state_dict["mlp_head_units"] = mlp_head_units
    
    # training parameters
    global_state_dict["train_batch_size"] = train_batch_size
    global_state_dict["val_batch_size"] = val_batch_size
    global_state_dict["val_steps"] = val_steps
    global_state_dict["epochs"] = epochs
    global_state_dict["steps_per_epoch"] = steps_per_epoch
    global_state_dict["augmentation"] = augmentation
    global_state_dict["crop"] = crop
    global_state_dict["crop_size"] = crop_size
    
    dataset_root, checkpoint_path, log_path, predictions_path = get_paths(root, dataset_root, checkpoint_root, log_root, time_stamp, architecture, classes)

    train_dataframe = DataFrame(dataset_root, train_catalog, classes, build_paths, binarised_labels, input_shape, input_dim, seed, ds_fraction, weight_mode, beta)   
    train_dataframe.show_image_label_pair(root, 3467, architecture, augmentation, crop, crop_size)
    weighting = train_dataframe.get_feature_weights()
    
    Model = Network(summary, architecture, learning_rate, weight_reg, activation_function, classes, input_shape, input_dim, loss_function, num_patches, projection_dim, num_heads, transformer_layers, mlp_head_units, log_path, checkpoint_path, save_step, saved_weights_path, global_epoch, global_batch, steps_per_epoch, weighting, predictions_path)
    train_data, validation_data = train_dataframe.get_train_val_split(train_dataframe.dataset, val_fraction) 
    Model.train(train_data, validation_data, train_dataframe, train_batch_size, val_batch_size, val_steps, epochs, augmentation, crop, crop_size)
    Model.save()
        
if __name__ == "__main__":
    """run `python train.py --help` for argument information""" 
    parser = argparse.ArgumentParser(prog = "train.py")
    
    # global parameters
    parser.add_argument("-s",           "--seed",                   type = int,         default = 0,                                                                                                    help = "(int) The seed for the random number generator")
     
    # model saving parameters
    parser.add_argument("-sdir",        "--save_dir",               type = str,         default = None,                                                                                                 help = "(str) The path to the checkpoint directory")
    parser.add_argument("-js",          "--json",                   type = str,         default = "meta.json",                                                                                          help = "(str) The path to the meta.json file containing the arguments")
    parser.add_argument("-swp",         "--saved_weights_path",     type = str,         default = None,                                                                                                 help = "(str) The path to the saved weights file")
    parser.add_argument("-ss",          "--save_step",              type = int,         default = 29298,                                                                                                help = "(int) The number of training iterations to wait before saving the model")
    
    # file paths
    parser.add_argument("-r",           "--root",                   type = str,         default = r"/home/malachy/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning",     help = "(str) The path to the root directory")
    parser.add_argument("-dsr",         "--dataset_root",           type = str,         default = r"galaxyzoo2-dataset-augmented/",                                                                     help = "(str) The path to the dataset root directory")
    parser.add_argument("-l",           "--train_catalog",          type = str,         default = r"gz2_train_catalog.parquet",                                                                         help = "(str) The name of the training catalog file")
    parser.add_argument("-tl",          "--test_catalog",           type = str,         default = r"gz2_test_catalog.parquet",                                                                          help = "(str) The name of the test catalog file") 
    parser.add_argument("-cpr",         "--checkpoint_root",        type = str,         default = r"checkpoints/",                                                                                      help = "(str) The path to the checkpoint directory")
    parser.add_argument("-log",         "--log_root",               type = str,         default = r"logs/fit/",                                                                                         help = "(str) The path to the log directory")
    
    # dataset parameters
    parser.add_argument("-c",           "--classes",                type = str,         default = "all-features",                                                                                            help = "(str) The classes to consider for each image, summarised as one 'representation'")
    parser.add_argument("-wm",          "--weight_mode",            type = str,         default = "effective_num_samples",                                                                              help = "(str) The method to use when calculating the class weights: 'inverse', 'inverse-sqrt', 'effective_num_samples', None")
    parser.add_argument("-bet",         "--beta",                   type = float,       default = None,                                                                                                 help = "(float) Constant used when determining the effective number of samples when class weighting 'effective_num_samples' is used, if None beta is calculated based on the total number of samples")
    parser.add_argument("-bp",          "--build_paths",            type = bool,        default = True,                                                                                                 help = "(bool) If to whether to build the paths to the images in the catalog file")
    parser.add_argument("-bi",          "--binarised_labels",       type = bool,        default = True,                                                                                                 help = "(bool) If to whether to binarise the labels")
    parser.add_argument("-frac",        "--ds_fraction",            type = float,       default = 1,                                                                                                    help = "(float) The fraction of the dataset to use, range: (0, 1]")
    parser.add_argument("-vf",          "--val_fraction",           type = float,       default = 0.2,                                                                                                  help = "(float) The fraction of the dataset to use for validation, range: (0,1]")
    
    # image properties
    parser.add_argument("-dim",         "--input_dim",              type = int,         default = 3,                                                                                                    help = "(int) The input dimension of the data")
    parser.add_argument("-ixy",         "--input_shape",            type = tuple,       default = (224, 224),                                                                                           help = "(tuple) The input shape of the data, i.e (424, 424) for 424x424 images")
    
    # network parameters
    parser.add_argument("-sum",         "--summary",                type = bool,        default = True,                                                                                                 help = "(bool) Display the summary of the network")
    parser.add_argument("-a",           "--architecture",           type = str,         default = "ViT",                                                                                                help = "(str) The architecture of the network")
    parser.add_argument("-lr",          "--learning_rate",          type = float,       default = 0.001,                                                                                                help = "(float) The learning rate of the network")
    parser.add_argument("-lf",          "--loss_function",          type = str,         default = "binary_crossentropy",                                                                                help = "(str) The loss function to use in the optimiser")
    parser.add_argument("-act",         "--activation_function",    type = str,         default = "sigmoid",                                                                                            help = "(str) The activation function of the final layer")
    parser.add_argument("-wr",          "--weight_reg",             type = float,       default = None,                                                                                                 help = "(float) The weight regularization of the network")
    
    # transformer model hyperparameters
    parser.add_argument("-patch",       "--num_patches",            type = int,         default = 256,                                                                                                  help = "(int) The number of patches to split the input image into")
    parser.add_argument("-projdim",     "--projection_dim",         type = int,         default = 64,                                                                                                   help = "(int) The dimensionality of the projection layer")
    parser.add_argument("-heads",       "--num_heads",              type = int,         default = 24,                                                                                                   help = "(int) The number of attention heads per patch")
    parser.add_argument("-tlayers",     "--transformer_layers",     type = int,         default = 12,                                                                                                   help = "(int) The number of transformer layers in the model")
    parser.add_argument("-mlps",        "--mlp_head_units",         type = list,        default = [2048, 1024],                                                                                         help = "(list) The number of units in the multi-layer perceptron head")
    
    # training parameters
    parser.add_argument("-bs",          "--train_batch_size",       type = int,         default = 16,                                                                                                   help = "(int) The batch size of the training data")
    parser.add_argument("-vbs",         "--val_batch_size",         type = int,         default = 16,                                                                                                   help = "(int) The batch size of the validation data")
    parser.add_argument("-vit",         "--val_steps",              type = int,         default = 512,                                                                                                  help = "(int) The number of validation batches to use per validation step")
    parser.add_argument("-e",           "--epochs",                 type = int,         default = 10,                                                                                                   help = "(int) The number of epochs to train the network")
    parser.add_argument("-iter",        "--steps_per_epoch",        type = int,         default = 58594,                                                                                                help = "(int) The number of iterations to train the network per epoch")
    parser.add_argument("-aug",         "--augmentation",           type = bool,        default = True,                                                                                                 help = "(bool) Whether to augment the images")
    parser.add_argument("-cro",         "--crop",                   type = bool,        default = True,                                                                                                 help = "(bool) Whether to centrally crop the images, if False the images are resized instead")
    parser.add_argument("-csize",       "--crop_size",              type = tuple,       default = (224, 224),                                                                                           help = "(tuple) The dimensions to crop the image to, must be the same as the input_shape to the network")
    
    args = parser.parse_args()
          
    if args.save_dir:
        json = os.path.join(args.save_dir, args.json)   
        main_kwargs = read_json(json)
        main(**main_kwargs)
    else:
        main_kwargs = args
        main_kwargs.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        main_kwargs.global_epoch = 0
        main_kwargs.global_batch = 0
        main_kwargs = vars(main_kwargs) # arg parse dict behaves like a dict in every case except in it's own class
        
        if main_kwargs["input_shape"] != main_kwargs["crop_size"] and main_kwargs["crop"] is True:
            raise ValueError("\033[31m'crop_size' must equal 'input_shape'!\033[0m")
            
        main_kwargs.pop("save_dir")
        main_kwargs.pop("json")
        main(**main_kwargs)