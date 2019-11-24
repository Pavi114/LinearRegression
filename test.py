from linearRegressionFromScratch import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


def show(x_test, Y_test, Y_predicted):
    fig, axes = plt.subplots(1, 2)

    axes[0].scatter(x_test, Y_test)
    axes[0].plot(x_test, Y_predicted, 'b-', label="Predicted")
    axes[0].set_xlabel('X')  # For this sample
    axes[0].set_ylabel('Y')
    axes[0].grid()

    axes[1].scatter(x_test.index, Y_test, color="green", label="Actual GPA")
    axes[1].scatter(x_test.index, Y_predicted, color="blue", label="Predicted GPA")
    axes[1].set_xlabel('Index of X')
    axes[1].set_ylabel('Y')
    axes[1].grid()

    plt.show()


def main():

    # Reading a sample data
    data = pd.read_csv('train.csv')
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1:]

    # Example data-set has separate train and test data
    X_train = X
    Y_train = Y

    data = pd.read_csv('test.csv')
    X_test = data.iloc[:, :-1]
    Y_test = data.iloc[:, -1:]

    # Splitting into test and train data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Training the model
    reg = LinearRegression(X_train, Y_train, 1500)
    reg = reg.fit()
    Y_predict = reg.predict(X_test)

    # plotting the results
    show(X_test, Y_test, Y_predict)


if __name__ == "__main__":
    main()

