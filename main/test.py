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
    crop_size,
    test_batch_size,
    predict,
    evaluate
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
   
    test_dataframe = DataFrame(dataset_root, test_catalog, classes, build_paths, binarised_labels, input_shape, input_dim, seed, ds_fraction, weight_mode, beta)   
    test_dataframe.show_image_label_pair(root, 1, architecture, augmentation, crop, crop_size)
    weighting = test_dataframe.get_feature_weights()
    
    Model = Network(summary, architecture, learning_rate, weight_reg, activation_function, classes, input_shape, input_dim, loss_function, num_patches, projection_dim, num_heads, transformer_layers, mlp_head_units, log_path, checkpoint_path, save_step, saved_weights_path, global_epoch, global_batch, steps_per_epoch, weighting, predictions_path)
   
    if predict:
        Model.predict(test_dataframe.dataset, test_dataframe, test_batch_size, augmentation, crop, crop_size)
    
    if evaluate:
        Model.evaluate(test_dataframe.dataset, test_dataframe, test_batch_size, augmentation, crop, crop_size) 
        
if __name__ == "__main__":
    """run `python test.py --help` for argument information""" 
    parser = argparse.ArgumentParser(prog = "test.py")
    
    # model saving parameters
    parser.add_argument("-sdir",            "--save_dir",               type = str,             default = None,                                     help = "(str) The path to saved model")
    parser.add_argument("-sbs" ,            "--test_batch_size",        type = int,             default = 32,                                       help = "(int) The batch size for testing")
    parser.add_argument("-test",            "--test_catalog",           type = str,             default = "gz2_test_catalog.parquet",               help = "(str) The name of the catalog file for the test dataset")
    parser.add_argument("-pred",            "--predict",                type = bool,            default = True,                                     help = "(bool) If to get predictions from the model on the test dataset")
    parser.add_argument("-eval",            "--evaluate",               type = bool,            default = False,                                    help = "(bool) If to evaluate the model on the test dataset")
    
    args = parser.parse_args()
          
    if args.save_dir:
        json = os.path.join(args.save_dir, "meta.json") 
        model_kwargs = read_json(json)
        model_kwargs.pop("test_catalog")
        kwargs = vars(args)
        kwargs.pop("save_dir")
        main(**kwargs, **model_kwargs)            
    else:
        raise ValueError("\033[31mYou must specify a model to test! \n\nHint: `python test.py --save_dir '<root>/checkpoints/<model>/<checkpoint>/'`\nrun `python test.py --help` for argument information.\033[0m")
            
        