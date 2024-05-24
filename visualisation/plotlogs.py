import argparse, os, csv  # noqa: E401
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    """
    Read a csv file from a specified path.
    
    args:
    path (str): path to the csv file
    
    returns:
    data (list): data contained within the csv file
    """
    data = []
    
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
            
    return data

def construt_x_values(data):
    """
    Build an array of the x axis to plot the data
    
    args:
    data (list): the data to plot
    
    returns:
    x (np.array): the x values to map the data to when plotting
    """
    return np.arange(0, len(data), 1)

def metrics_to_plot(metric_log, figsize, dpi, line_colour, x_label, y_label, title):
    """
    Plot a specified metric csv using matplotlib.
    
    args:
    metric_log (str): the path to the csv file containing the desired metric to plot
    line_colour (str): the desired line colour for the data
    x_label (str): the desired x label to plot
    y_label (str): the desired y label to plot
    title (str): the desired title to plot
    """
    data = read_csv(metric_log)
    x = construt_x_values(data)
    
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    ax.plot(x, data, color = line_colour, linewidth = 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()
    
def main(root, log_dir, log, accuracy, loss, rms_error, val_accuracy, val_loss, fig_size, dpi):
    
    log_path = os.path.join(root, log_dir, log)
    
    if accuracy:
        accuracy_csv = os.path.join(log_path, "accuracy.csv")
        metrics_to_plot(accuracy_csv, fig_size, dpi, line_colour = 'orange', x_label = 'Batch', y_label = 'Accuracy', title = 'Accuracy per Batch')
        
    if loss:
        loss_csv = os.path.join(log_path, "loss.csv")
        metrics_to_plot(loss_csv, fig_size, dpi, line_colour = 'blue', x_label = 'Batch', y_label = 'Loss', title = 'Loss per Batch')
        
    if val_accuracy:
        val_accuracy_csv = os.path.join(log_path, "val_accuracy.csv")
        metrics_to_plot(val_accuracy_csv, fig_size, dpi, line_colour = 'magenta', x_label = 'Epoch', y_label = 'Validation Accuracy', title = 'Validation Accuracy per Epoch')
        
    if val_loss:
        val_loss_csv = os.path.join(log_path, "val_loss.csv")
        metrics_to_plot(val_loss_csv, fig_size, dpi, line_colour = 'cyan', x_label = 'Epoch', y_label = 'Validation Loss', title = 'Validation Loss per Epoch')
        
    if rms_error:
        pass
        
if __name__ == "__main__":
    """run `python plotlogs.py --help` for argument information"""
    parser = argparse.ArgumentParser(prog = "plotlogs.py")
    
    # paths
    parser.add_argument("-r",       "--root",           type = str,         default = "/home/malachy/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/", help = "(str) The path to the root directory")
    parser.add_argument("-dir",     "--log_dir",        type = str,         default = "logs/fit",     help = "(str) The path containing the log files")
    parser.add_argument("-log",     "--log",            type = str,         default = None,           help = "(str) The log file to use")
    
    # metrics
    parser.add_argument("-a",       "--accuracy",       type = bool,        default = True,           help = "(bool) If to plot the Accuracy over batches")
    parser.add_argument("-l",       "--loss",           type = bool,        default = True,           help = "(bool) If to plot the Loss over batches" )
    parser.add_argument("-rmse",    "--rms_error",      type = bool,        default = True,           help = "(bool) If to plot the Root Mean Squared Error over batches")
    
    parser.add_argument("-va",      "--val_accuracy",   type = bool,        default = True,           help = "(bool) If to plot the Validaiton Accuracy over epochs")
    parser.add_argument("-vl",      "--val_loss",       type = bool,        default = True,           help = "(bool) If to plot the Validation Loss over epochs")
    
    parser.add_argument("-fg",      "--fig_size",       type = tuple,       default = (7, 4),        help = "(tuple) The figure size for the plot(s)")
    parser.add_argument("-d",       "--dpi",            type = int,         default = 300,            help = "(int) The dpi for the plot(s)")
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    if kwargs["log"] is None:
        raise ValueError("\033[31mPlease provide a valid log folder!\033[0m")
    
    main(**kwargs)
    