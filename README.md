# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.



<img width="1893" height="994" alt="image" src="https://github.com/user-attachments/assets/84632660-22d4-40ed-a7f1-597891b815c1" />




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
```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.model(x)

model = PeopleClassifier(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
print("Name: Shivasri")
print("Register Number: 212224220098")
print("\nTraining Output\n")

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

cm = confusion_matrix(y_test, predicted)

print("\n\nConfusion Matrix")
print("Name: Shivasri")
print("Register Number: 212224220098\n")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report")
print("Name: Shivasri")
print("Register Number: 212224220098\n")
print(classification_report(y_test, predicted))

# -----------------------------
# New Sample Prediction
# -----------------------------
print("\nNew Sample Data Prediction")
print("Name:Shivasri S")
print("Register Number: 212224220098\n")

# Example sample (change values if needed)
sample = X_test[0].unsqueeze(0)

prediction = model(sample)
_, predicted_class = torch.max(prediction, 1)

segments = ["A", "B", "C", "D"]

print("Input Sample:", sample.numpy())
print("Predicted Segment:", segments[predicted_class.item()])
```


## Dataset Information

Include screenshot of the dataset

<img width="1919" height="928" alt="image" src="https://github.com/user-attachments/assets/30ba02d4-8af5-4369-9af0-ff6ec5c98cee" />




## OUTPUT
<img width="928" height="358" alt="image" src="https://github.com/user-attachments/assets/e09130ce-13f3-40b3-910a-2700a7147471" />

### Confusion Matrix

Include confusion matrix here

<img width="835" height="694" alt="image" src="https://github.com/user-attachments/assets/2e67e979-1e9d-4471-ba97-d073d61dda3a" />




### Classification Report

Include Classification Report here


<img width="720" height="346" alt="image" src="https://github.com/user-attachments/assets/5fcc399c-af1e-4019-ae09-e25cf1de2888" />



### New Sample Data Prediction



<img width="1615" height="169" alt="image" src="https://github.com/user-attachments/assets/0efcda86-c048-4739-b7bb-459d79092190" />


### Result:

The Neural Network model was successfully developed to classify customers into four segments (A, B, C, D) based on Age, Annual Income, and Spending Score.




## RESULT
The Neural Network Classification Model was successfully developed and trained using PyTorch.
