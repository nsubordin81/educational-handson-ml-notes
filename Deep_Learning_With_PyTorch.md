# Deep Learning With Pytorch

## Tensors

assertion, pytorch is more coherent with the way you code in python as well as the concepts of deep learning. in tensorflow and keras, instructor says you have to code in a way that doesn't match deep learning as well. 

talk about tensors: 

They are a generalization of vectors and matrices, vectors being 1d line of values, standard matrix being a 2d tensor, tensor can be higher dimension than this too, like stacks of matrices. it gets easier if you can visualize them in your mind as you work with them so you can visualize how they work and how they flow through the network you build. 

tensors are the basic data structure in pytorch, it is good to understand how they work. 

this mostly works like numpy arrays, you can also slice them and index them in the same way. .size() is useful becauas you can get the dimensions of a tensor and you can make another tensor off of that one. 

very important is that pytorch lets you have two optionsfor operations, either a mutating version or an immutable version. underscore means you are using mutation. 

important methods, you can check the size or shape of a tensor you use.size()
if you want ot resize, you use resize with an underscore and then you can specify a new size. the reshaping is done as an in place method

numpy arrays can really easily be converted back and forth to torch tensors. 
you just need to create a random array with numpy then you can change the array to a torch tensor with from_numpy
if you have a tensor you can use .numpy to turn it back into a numpy array

the tensor and the numpy array will share memory, so if you multiply a tensor in place and then look back at a numpy array that we converted the tensor into, the two references will be mutating each other. 

## Defining Neural Networks With Pytorch

imports common, torch for pytorch, numpy, and torchvision which lets you get some existing datasets. 

first you use transform to normalize the image using a Comple method. you call ToTensor on images to make them into a tensor, then you normalize them with the Normalize function to transform the value space for the values from 0,1 to -.5,.5 and then /.5 so you end up with a domain from -1 to 1. 

the pytorch Dataloader class lets you set up the batch torch.utils.data.Dataloader you pass in a batch size and a shuffle option to have the minibatches selected a piece at a time. 

### network dimensionality

start with 784x28x28, hidden layer 128 units relu activation, hidden layer 64 units, output layer 10 units and uses softmax layer. then you use a loss layer for training

softmax converts to discrete probability distribution, then the loss layer compares the output fo the softmax to the true label. 

torch has nn, neural network functions
nn.functional is F

typicall way you use pytorch to define a neural network is to define a class that inherits from nn.Module, then call super in the instructor, which will call the init method of nn.Module. 

you define the layers in the init function and then you create a forward function to pass a tensor through each of your layers, your linear series of operations

you take that tensor and then apply the relu activation and then go throug the next layer, keep doing that in turn and then do your softmax function at the end. 

the convention is that your things that are transformations that you define will be in init and then softmax and relu your activations will be in forward. So to remember your first dimension, dimension zero is the one that is how many examples you have, and then dimension one is the dimension of the vector tha tyou are passing through. 

when you create the layers, the weights and biases are automatically created and initialized. However they are initialized to random values. so you can get them with model.layername.bias.data or model.layername.weight.data. you can then intialize them to zero or a normal distribution

trainloader returns a generator, then you call iter onit to getan interator, then you call next to get the next value

images come as 28x28 images and you want ot convert them to a flattened 784 dim vector. you can use resize here to start with 64, 1, 748 and that will resize it to a 748 column vector. you can asol reference the number of rows without hardcoding it using the shape dimensions of the tensor

then you can just call forward to get back the probabilities, your class scores 
there is a helper file that was made for the course that alllws you to plot an image and view how the network classified the image. To use the class they wrote theough you have to use a view method, which is like resize but it is immutable operation so it returns a new tensor that is the resized version fo the origianl. 

there is another way to build a model called nn.sequential. 
you first define the parameters and hyper parameters, 
then you pass in the transformations that you would like to do, you don't have to name them. so you might do nn.Linear and then relu and the nlinear and then relu and then linear and then softmax, all the time using the parameters you defined. It is a more functional way to define the network

for sequential you can also pass in an ordered dict, so you pass in a list of tuples that are names and values, so you name the layers as the keys and you use the values to define the layres themselves. The keys must be unique. 

### Training Neural Networks In Pytorch

interesting characterization of gradient descent, it will 'always go in the direction of fastest change' and that is why it quickly finds the minima. you find out the gradient of the loss with respect to the weights, so the loss function gives you the total loss and then you find the partial derivatives of the loss wirht respect to each of the weights moving backward through the network along the path that gets you to that weight, and use the chain rule to get the product of the gradient in each step which will give you the gradient with respect to the weights

criterion is what you often call the loss function in pytorch. Then there is an optimizer which uses the gradients that are found by gradient descent to update all the weights in the network. 

### Autograd

if you have a tensor and you tell it that it requires a gradient, then autograd will track all the operations that happen on the tensor, then you use z.backwrad at the end. it can then go backwards and calculate all the gradients that you want to use. 

you have to tell the tensor when you create it with requires_grad=True

you can test it with simple tensor operations, like squaring your tensor and storing the results in a new variable, then you canlook at the grad_fn of the tensor and see the history of operations applied to the tensor

You still have to do a backward pass through the operations to calculate the gradient after doing forward pass. use the backward() method to calculate the gradients of one tensor with respect to another. 

those are the ingredients to training
1. you do a forward pass through the network
2. then you define a loss function and optimizer
3. output, logits, softmax, etc. to calculate the loss
4. calculate the gradients with respect to the loss with your criterion loss.backward()
5. then you can pass the gradients to a step that the optimizer takes to update the weights. the tensors have them so you don't have to pass them actually
6. important note, every time you do the call to backward, you will be accumulating the gradients, so you should zero up grad for your optimizer and only call backward once per iteration of the model.
7. call step() on the optimizer in order to apply the gradients to the weights and biases

you first want to go through your epochs which is one pass through the entire data set. you will then keep tabs on your running loss amount you want that to be ssteadily decreasing as you go through the training process. 

so your outer loop is your epochs and your inner loop is every element in your minibatch

your running loss is going to be a scalar tensor whihc is really just a number but there is a convenience method to get the value out of the tensor which is loss.item()


## Inference and Validation

neural networks tend to perform too well on training data adn aren't able to generalize to other data, this is overfitting, we went over this in the neural networks and CNN parts of the course. 

you use the hold out set for validation either called validation or test set, yes this is review. torchvision provides validation datasets. 

one point that was covered here that wasn't covered in the training of the fashion MNIST in the CNN part of the lesson was that another thing that waas done to improve model accuracy was to take the log of softmax instead of just doing softmax. The reasoning behind this is that the softmax ittself is probably going to have a lot of probabilities that are very close to but not 1 and very close to but not 0, and because of floating point representation on computers you can run into issues with these very small differences in decimal representation. So the solution is to take the log of the softmax because that will transform or normalize the number space into a range that is easier to work with. 

one thing to consider that I didn't know when doing my CNN example was that the cross entropy loss function criterion that I chose expects that the forward pass output was the softmax range, and the negative log likelihood loss function instead will expect the log of the softmax, so I should probably have switched to that one. 

also, Adam is more about faster than it is about better? sounds like it was what the video said. 

validation code pseudo 
perform inference test on the test data
measure the loss on the test data
measuer the model accuracy on the test data

get images and labels from the testloader iterator
reshape them into vectors instead of 2d 

get the output from the forward pass 
pass that into the criterion directly with the labels adn get the loss value and keep a cumulative sum of the test loss for each one. 

interesting step takine here you know the output is the log of the softmax so if you want to get back to softmax to see what the class scores were then you take that  output to an exponent. 

you can then take the max of the class scores, then your max value from torch tensor will return 2 tensors, one is the value of the max value, and the second tensor is telling you the index in the output vector of those max values. 

you create an equality tensor by comparing the predicted class which is the second in the max with the true labels. you end up with an array where 1 is correct and 0 is incorrect

accuracy is how often we were right, so that is just the mean of that vector that is all ones and zeros. If you run into the problem that mean isn't on the tensor you want then you could change the tensor into a different type just do .typeand then pass the torch.<type> of t tensor into that type function adn then you hvae access to the mean. 

you want to do this function in a loop, so you can make the whole validation thing a function and loop through all the mi nibatches and calculate the test loss and accuracy. 

then, in a context manager with torch.no_grad: which turns off all the gradients for the tensors inside of it, you can switch between model.eval() and model.train() modes so that when you are doing training the validation code won't run and then when you are evaluating it will run. 

you can then set up the trained model for inference using the model.eval() mode so that the dropout code doesn't run and you can turn off gradients because you aren't doing gradient descent, and then you can do inference steps and check the probabilities you get from your forward pass to look manually at how well you are classifying specific examples. 

## Saving and loading models

don't need to train a model every time you want to use it
new import
import fc_model, you are directly importing your model that you created in an earlier step. 

the parameters are stored in an attribute called state_dict
you can look at the model and print out the keys of your state_dict and it shows you that you have hidden layers and the weights and biases of your model that is what you are essentially saving, the trained weights and biases of your model. They get saved to dict. 

torch.save you pass in the state_dict then you also pass a file path for the checkpoint

once you want to reload later you can load it with model.load_state_dict(state_dict)

there is a gotcha here, if you want to load a state dictionary into a new model you have created where the checkpoint of the weights and biases don't have the same dimensions (architecture) of the model that you are trying to load into. So you need to save the dimensions of the model as well as the weights and biases. 

you can specify the sizes of the input size the output size and the shapes of each of the hidden layers in a checkpoint dictionary and you also should include a state_dictionary with the dictionary of weights and biases in it as a key in this chekcpoint dictionary. 

you can use this same torch.save checkpoint to save it. when you do this, loading and creating the model you don't have to specify the architecture of the model because you already have it stored into the checkpoint dictionary

### loading image data

you can use dataset.ImageFolder(path/to/data, transform=transforms)
you have a folder for each of the different classes or labels that you want ot train for and then you have each of the imgaes for those classes inside, so it is based on the directory naming structure. 

you don't necessarily have all the image processed the way you want when youload them in, so there are transforms that pytorch provides that will allow you do to things such as resize, crop, convert the data to a tensor, etc. these transforms can be composed together to result in a kind of pipeline of image transforms. 

you then can take your dataset and pass it to the dataloader, and you can specify how big your batches are and whether or not you should shuffle the examples as they come in from the batches, this is preferred most of the time to preven you having some correlation in the order the model sees the images for things like models that learn a sequence to have an impact on the training of the model. 

your dataloader is a generator so you have to do Next(iter(data(dataloader)) to get data out of it one by one or you can use a loop to get data out of the generator one by one and the iterator stuff is hidden for you conveniently. 

there are also tools that help you with data augmentation that will make the model more robust to images that have different distortions applied to them. The validation set shouldn't get the augmentations so you really only want ot do this on the training data. your test examples should not look like tha tbecause they should be closer to what you would actually send into your model to predict on so they should be centered. 

The cat dog images task that was previewed in this exercise is actually hard to build a good network for, it is more qadvanced because it is classifying using a larger diversity of images for training that haven't been carefully chosen from a standardized set like those of mnist and fashion mnist that are more used for evaluating baseline performance of models. As a result, this is probably going to be a higher lift of effort to get right and you might be reinventing the wheel, so why not do some transfer learning. 


## transfer learning

I watched while away from keyboard but my notes were that generally you can create a classifier that reads in the features from the output of the layer just prior to softmax in the original, very deep netowrk that has already been trained on a ton of data. You can then leverage the highly improved performace that this model has been able to achieve with the massive amount of data it was trained with. they created another layer that took in the output of that prior layer as its input dimension and then pass that through a fully connected layer which reduced the dimensions, then an activation for that relu, and then another fully connected layer that reduced the dimensions to the number of classes for the transferred set, and then a softmax on that to convert it to a probability distribution in terms of the clsses they wanted to learn. Another important thing that was done priori to adding this classifier, they froze the parameters of all the layers of the network that were being used pretrained, by basically turning requires grad to false for all parameters by looping through the parameters list. that gets us to pretty much where we want to be for this. Then you set the clssifier attribute for that model to waht you created insteado f using the softmax layer that the model already had. Oh by the way the imagenet trained model that was used was taken from the pytorch models you can actually import it directly. 

now that you aare doing transfer learning with a really deep model, you should use CUDA to train the model instead of relying on the cpu to train it you can easily switch between cpu and cuda using the model.to() function off of a model instance that inherits from the pytorch parent class.
