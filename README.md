<img width="1036" height="420" alt="image" src="https://github.com/user-attachments/assets/5960ca3c-8643-4237-9d92-4e169674135d" /># Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: 
### Register Number:

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
```python

# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information

Include screenshot of the dataset


<img width="1041" height="417" alt="image" src="https://github.com/user-attachments/assets/7c454a84-14b7-46d7-81de-7c47c89a70d8" />


## OUTPUT


### Confusion Matrix

Include confusion matrix here

<img width="839" height="713" alt="image" src="https://github.com/user-attachments/assets/0518cc64-dc13-4a35-828d-f67e51f1aec2" />


### Classification Report

Include Classification Report here
<img width="712" height="297" alt="image" src="https://github.com/user-attachments/assets/af4ba34d-5835-4d32-ad38-5cd6ca6a6062" />


### New Sample Data Prediction
<img width="970" height="322" alt="image" src="https://github.com/user-attachments/assets/3b83fcff-70bc-4991-bb2a-41e21a9f2c59" />



Include your sample input and output here

## RESULT
Include your result here
