% Load MNIST dataset 
[train_images, train_labels] = digitTrain4DArrayData;
[test_images, test_labels] = digitTest4DArrayData;

% Normalize the images 
train_images = double(train_images) / 255; 
test_images = double(test_images) / 255;


train_labels = categorical(train_labels);
test_labels = categorical(test_labels);



% Define the MLP architecture
layers = [
    imageInputLayer([28 28 1], 'Name', 'input')        
    fullyConnectedLayer(128, 'Name', 'fc1')            
    reluLayer('Name', 'relu1')                         
    fullyConnectedLayer(64, 'Name', 'fc2')             
    reluLayer('Name', 'relu2')                         
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(32, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(10, 'Name', 'fc5')
    softmaxLayer('Name', 'softmax')                    
    classificationLayer('Name', 'output')              
];

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 512, ...
    'MiniBatchSize', 1024, ...
    'ValidationData', {test_images, test_labels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(train_images, train_labels, layers, options);