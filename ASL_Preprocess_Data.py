from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from ASL_Folder_Init import *

#store in a map, maps label (hello, thanks, ily) to numerical values (1, 2, 3)
label_map = {label:num for num,label in enumerate(actions)}
# for label, numerical_value in label_map.items():
#     print(f"Action: {label}, Numerical Value: {numerical_value}")

# Lists to store sequences and corresponding labels
sequences, labels = [], []

# Loop through each action (gesture) in the 'actions' array
for action in actions:
    # Loop through a specified number of sequences for each action, no_sequences = 30 videos of data we collected
    for sequence in range(no_sequences):
        # List to store frames for the current sequence
        window = []
        # Loop through each frame in the sequence
        for frame_num in range(sequence_length):
            # Load the numpy array data for the current frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # Append the frame data to the 'window' list
            window.append(res)
        
        # Append the sequence of frames to the 'sequences' list
        sequences.append(window)
        # Append the numerical label corresponding to the action to the 'labels' list
        labels.append(label_map[action])

# np.array(sequences).shape = (90, 30, 258) = (30 'hello'+ 30 'thanks'+ 30 'iloveyou', 30 frames each, 258 features per frame)

#store it in a X matrices, same output shape = (90, 30, 258)
X = np.array(sequences)

#One-hot encoding: binary vector where all elements besides the index represents = 0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] since we have 3 labels
y = to_categorical(labels).astype(int)

#splits 95% train 5% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) #y_test.shape = (5, 3) since we trained 5% there is 1 row per X_test


def Evaluation():
    # Load the saved model
    model = load_model('action.h5')
    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist() 

    # Calculate and print confusion matrix
    cm = multilabel_confusion_matrix(ytrue, yhat)
    print("Confusion Matrix:")
    print(cm)

    # Calculate and print accuracy score
    acc_score = accuracy_score(ytrue, yhat)
    print("\nAccuracy Score:", acc_score)

def LSTM():
    #set logs file
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    #Deep learning
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    res = [.7, 0.2, 0.1]
    actions[np.argmax(res)]
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
    model.summary()
        
    res = model.predict(X_test)
    print(actions[np.argmax(res[4])])
    print(actions[np.argmax(y_test[4])])

    model.save('action.h5')

Evaluation()