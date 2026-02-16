# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

<img width="1880" height="983" alt="image" src="https://github.com/user-attachments/assets/b766cdba-8dfc-4a15-9111-a907f6726696" />


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
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



```

```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)


```
```python
# Confusion Matrix
cm = confusion_matrix(y_test, predicted)

print("\nName: Shivasri")
print("Register Number: 212224220098")

print("\nConfusion Matrix:")
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


```


## Dataset Information

Include screenshot of the dataset

<img width="1919" height="1022" alt="image" src="https://github.com/user-attachments/assets/b79115d2-948f-4368-ad15-40fa9711183c" />



## OUTPUT
<img width="1033" height="490" alt="image" src="https://github.com/user-attachments/assets/c8bfd1ca-150f-4cf7-861d-2b18d896356d" />


### Confusion Matrix

Include confusion matrix here

<img width="994" height="723" alt="image" src="https://github.com/user-attachments/assets/f97f7901-131a-4496-bc35-2a88038cede4" />



### Classification Report

Include Classification Report here


<img width="685" height="323" alt="image" src="https://github.com/user-attachments/assets/6a9fe531-bd7a-4d4b-b557-7c30283fb0a2" />


### New Sample Data Prediction


<img width="819" height="235" alt="image" src="https://github.com/user-attachments/assets/391b0962-e5f1-42f0-92e4-5212d66fda2c" />


### Result:

The Neural Network model was successfully developed to classify customers into four segments (A, B, C, D) based on Age, Annual Income, and Spending Score.




## RESULT
The Neural Network Classification Model was successfully developed and trained using PyTorch.
