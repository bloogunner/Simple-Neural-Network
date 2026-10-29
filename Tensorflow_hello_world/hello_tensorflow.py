import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

''' 
Training data: learn y = 2x + 1 | y^  = wx + b
y = 2(1) + 1 = 3 
y = 2(2) + 1 = 5 
y = 2(3) + 1 = 7 
y = 2(4) + 1 = 9
y = 2(5) + 1 = 11
We are training our model to understand this simple funciton.
'''
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([3, 5, 7, 9, 11], dtype=np.float32)




'''
Get the folder containing this Python script.
'''
folder = Path(__file__).resolve().parent

'''
Define the model's expected location.
'''
model_path = folder / "my_model.keras"

'''
Check whether the file exists.
'''
if model_path.is_file():
    print(f"Existing model found: {model_path}")
    model = tf.keras.models.load_model("my_model.keras")
else:
    print("No existing model found.")
    
    ''' 
    Create a simple neural network
    1. Model is our instantiated object for our neural network. 
    2. tf.keras.Sequential creates a nueral netowrk whose layers are arrange in order. Data enters the first layer,
    passes through the next layer, and eventually produces an output.
    3. tf.keras.Input(shape=(1,)) tells the model that each example contains one input number. 
    4. tf.keras.layers.Dense(units=1) creates a dense, or fully connected, layer with one output unit (one neuron).
    '''
    model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)])
    

    ''' 
    Configure the model
    1. Optimizer is us selecting which training model in this instance it is stochastic gradient descent.
    2. loss measures how wrong the models predictions are using mean squared for this example.
    (8-11)^2 = 9 is a loss of 9 
    (10-11)^2 = 1 is a loss of 1 so the model will use this prediction in this scenario
    ''' 
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="mean_squared_error"
    )

    # Set up a live visualization
    plt.ion()

    fig, (ax_graph, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Watching a Neural Network Learn")

    # Left: training data and the model's predictions
    ax_graph.scatter(x, y, color="green", label="Correct answers")
    line, = ax_graph.plot([], [], color="blue", label="Model prediction")

    ax_graph.set_xlim(0, 6)
    ax_graph.set_ylim(0, 14)
    ax_graph.set_xlabel("Input (x)")
    ax_graph.set_ylabel("Output (y)")
    ax_graph.legend()
    ax_graph.grid(True)

    # Right: training loss
    loss_line, = ax_loss.plot([], [], color="red")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean squared error")
    ax_loss.set_title("How wrong is the model?")
    ax_loss.set_yscale("log")
    ax_loss.grid(True)

    loss_history = []
    epochs_seen = []
        
    class LivePlot(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Update every 5 epochs to keep the display responsive
            if epoch % 5 != 0 and epoch != 499:
                return

            # Read the neuron's current weight and bias
            weights, bias = self.model.layers[0].get_weights()
            w = float(weights[0][0])
            b = float(bias[0])

            # Calculate predictions for drawing the line
            x_line = np.linspace(0, 6, 100, dtype=np.float32)
            y_line = w * x_line + b

            line.set_data(x_line, y_line)

            # Record the current loss
            loss = logs["loss"]
            loss_history.append(loss)
            epochs_seen.append(epoch + 1)

            loss_line.set_data(epochs_seen, loss_history)

            ax_loss.set_xlim(1, max(10, epoch + 1))
            ax_loss.relim()
            ax_loss.autoscale_view(scalex=False, scaley=True)

            # Show the current values in the title
            ax_graph.set_title(
                f"Weight: {w:.3f} | Bias: {b:.3f}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)
            
    ''' 
    Train the model
    Epochs is how many times we run through data set
    Verbose shows training progression based on its value
    0 shows no training progress
    1 shows a progress bar
    2 prints a summary for each epoch
    ''' 
    model.fit(x, y, epochs=500, verbose=0, callbacks=[LivePlot()])


    plt.ioff()
    plt.show()

'''
Users # for model to predict
'''
user_num = input("What number would  you like the model to predict?")


''' 
Asking the model to predict a new input that it has yet to see. 
We insantiate our object prediction for model.predict. 
We pass in a new value into the numpy array. 

''' 
prediction = model.predict(
    np.array([[user_num]], dtype=np.float32),
    verbose=1
)


'''
We print the prediction by calling the front of the arrays using our object predicition.
'''

print(f"Prediction for x={user_num}:", prediction[0][0])


'''
Print every tainable weight and bias.
'''

for variable in model.trainable_variables:
    print(variable.name)
    print(variable.numpy())
    

'''
Save the model for later use.
'''

model.save("my_model.keras")
