import pandas as pd
import matplotlib.pyplot as plt


def prepare_data(main_file, target_file, time_window):
    # Read main.csv and target.csv files into pandas dataframes
    main_df = pd.read_csv(main_file)
    target_df = pd.read_csv(target_file)

    # Extract 'W' values from main_df as feature matrix X
    X = []
    y = []
    for i in range(time_window, len(target_df)):
        timestamp = target_df['timestamp'][i]
        window_start = target_df['timestamp'][i - time_window + 1]
        window_end = timestamp
        window_df = main_df[(main_df['timestamp'] >= window_start) & (main_df['timestamp'] <= window_end)]
        window_w = window_df['W'].tolist()

        if len(window_w) != time_window:
            continue

        X.append(window_w)
        # Extract 'W' values from target_df as target variable y
        y.append(target_df[target_df['timestamp'] == window_end]['W'].values[0])

    X = pd.DataFrame(X)
    y = pd.Series(y)

    return X, y


def visualize_data(file_path):
    # Read the .csv file into a pandas dataframe
    df = pd.read_csv(file_path)

    # Extract the 'W' values
    w_values = df['W'].values

    # Create a time series plot of the 'W' values
    plt.plot(w_values)
    plt.xlabel('Timestamp')
    plt.ylabel('W')
    plt.title('W values over Time')
    plt.show()


def visualize_predictions(y_pred, y_true):
    """
    Visualizes predicted values against true values using a plot.

    Parameters:
        - y_pred (array-like): Predicted values.
        - y_true (array-like): True values.
    """
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('W')
    plt.legend()
    plt.show()


path_main = "data/1.csv"
path_target = "data/3.csv"
window_size = 10

if __name__ == '__main__':
    X, y = prepare_data(path_main, path_target, time_window=window_size)
    pass
