# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.
<img width="769" height="494" alt="image" src="https://github.com/user-attachments/assets/ed31850b-80db-46d2-a958-78d7b18d8340" />

## DESIGN STEPS

### STEP 1: Data Collection and Loading

Load the customer dataset (customers.csv) containing features like Age, AnnualIncome, SpendingScore and the target variable Segment.

### STEP 2: Data Preprocessing

Encode the target column (Segment) into numeric form, normalize the feature values, and split the dataset into training and testing sets.

### STEP 3: Model Construction

Define a Feedforward Neural Network using input layer, hidden layers with ReLU activation, and an output layer with 4 neurons for classification.

### STEP 4: Model Training

Initialize the loss function (CrossEntropyLoss) and optimizer (Adam), then train the model for a fixed number of epochs.

### STEP 5: Model Evaluation and Prediction

Evaluate the trained model using Confusion Matrix and Classification Report, and test it with new sample data for prediction.


## PROGRAM

### Name: Shivasri S
### Register Number: 212224220098

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)  # 4 segments
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


```
```python

input_size = X_train.shape[1]

model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



```
```python
def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    losses = []
    
    for epoch in range(epochs):
        model.train()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return losses

losses = train_model(model, X_train, y_train, criterion, optimizer, epochs=100)

```



## Dataset Information

Include screenshot of the dataset

<img width="1919" height="1022" alt="image" src="https://github.com/user-attachments/assets/b79115d2-948f-4368-ad15-40fa9711183c" />



## OUTPUT
<img width="1033" height="490" alt="image" src="https://github.com/user-attachments/assets/c8bfd1ca-150f-4cf7-861d-2b18d896356d" />

<img width="1509" height="712" alt="image" src="https://github.com/user-attachments/assets/b9d7264a-b53f-499f-9948-515b8d649cd8" />

### Confusion Matrix

Include confusion matrix here

<img width="761" height="301" alt="image" src="https://github.com/user-attachments/assets/8d539848-4683-456a-ba00-fd8224b94107" />


### Classification Report

Include Classification Report here
<img width="746" height="295" alt="image" src="https://github.com/user-attachments/assets/d15b41f4-3dae-43b3-aa91-0d77c29f534c" />



### New Sample Data Prediction

<img width="1097" height="409" alt="image" src="https://github.com/user-attachments/assets/0060cda1-016c-4346-92e3-4e9b4d7d9923" />

### Result:

The Neural Network model was successfully developed to classify customers into four segments (A, B, C, D) based on Age, Annual Income, and Spending Score.


Include your sample input and output here

## RESULT
Include your result here
