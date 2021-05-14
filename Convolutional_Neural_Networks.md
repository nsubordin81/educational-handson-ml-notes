# Convolutional Neural Networks

 convolutional neural netowrk is a deep neural network that is best at image processing tasks like classifying images by object type. they have layers that process visual information. They take in input images and then pass them through each of the different layers and there are different kinds of layers they go through. Three common subtypes of convolutional layers are convolutional, pooling, and fully connected. 

Example of VGG-16 for complete architecture, what does it look like? each layer is going to be represented by its dimentionality. 

image goes in, class for image comes out. 

224x224x3 image, first 2 layers of convolution and ReLU at 224x224x64, then 1 max pooling at 112x112x128 followed by 2 more convolution + relu at the same, then another max pooling but at 56xc56x256 then 3 more convolutional + ReLU at same, then 1 max pooling at 28x28x512 then 3 more convolutional + ReLU at the same, then 1 more max pooling  at 14x14x512 followed by 3 more convolutional and ReLU at same, then a 7x7x512 max pooling layer followed by 2 fully connnected with ReLU at 1x1x4096 and then 1 fully connected + ReLU at 1x1x1000 and the final softmax at that same dimension, so you get a probability that the image is one of 1000 different types of objects. 

## Convolutional Layer

1. image is the input. 
2. The layer is composed of convolutional filters. each of the convolutional filters extracts a specific kind of feature, like a high pass filter to detect the edges of the object. 
3. the output of the convolutional layer is the set of feature maps (these are also cknoen as activation maps) that are filtered versions of an original input image. 

Convolutional layers use a Rectified Linear Unit for their activation. the description given activatoin functions from this course is that they are placed after a convolutional layer to "slightly transform the output so that it's more efficient to perform backpropagation and effectively train the network."

Description of convolutional filters. you have a region of an image where you might want to recognize several different features. You can have many filters in one convolutional layer (can be 10s to 100s) which are convolving over the image data with their own sets of weights which allow them to detect some diffeent set of patterns when they pass over a subsection of the image. 

example, there is a udacity car image, 4 filters are applied, they are each 4 pixels tall. they will be convolved across the height and width of the image to produce a collection of the nodes in the convolutional layer. So there is something I hadn't visualized yet. So each convolutional filter corresponds to a collectoin of nodes in that layer. So the neural net is still learning a series of functions as a linear combination of the outputs of the prior layer (in this case it is the image features in pixes) by different sets of weights and biases applied to those features, So that is what each of those neurons represent, however, these functions also represent the convolutions of a filter across the image and the resulting matrix that it ends up with after convolution. 

so the resulting node collections are 'feature maps' or 'activation maps'. when you visualize the feature mapp you can see that they actually discover different features in the image, and you can see that the type of features the filters identify correspond to the way they are set up, you can purposefully create filters that identify edges in the image either as horizontal or vertical edges. 

color images are represented as a 3d tensor, a '3d array' that has r, g and b layers. you still will folow the process, so you still move the filter across and down the image, you convolve a 3d filter over the input color image which is also represented as 3d rgb matrix. however, since you don't want to have separate nodes in your network for each color channel, you would want to sum those up to become one output set of nodes that represents the sum of your red, green and blue channels for one filter. But then, something cool you can do after that is do that same process of combining that you did for color channels and apply it to the different filters you came up with and then pass that combined set of filters into another convolutional layer in order to discover patterns in the patterns that the convolutional filters you ended up with. 
This nesting process of pattern discovering could be repeated as many times as it takes. 

differences in convolutional layers from fully connected layers. hwen you think of neural net you usually think of them as the fully connected layers where you have weights going from every node in the prior layer to every node in the following layer. But in convolutional layers this is not the case. the nodes in a convolutional layer are restricted in their connection to only a small subset of the nodes in the prior layer. Convolutional layers also share their parameters, but both fully connected and convolutional layers their are weights and biases that are intially randomly generated. Filters and patterns are also intially randomly set to patterns that are random, so the CNN learns what patterns are appropriate based on its gradient descent against the loss function. There is also alwaays a loss function. Multi Class classification uses multi class cross entropy loss. Just like any neural network we are tryinig ot have the data provide the way to get to the function thgrough the optimization problem being solved, minimizing the preexisting bias we introduce into the model, so CNNs aren't made to just figure out how to recognize one type of object with preset filters. That would be a lot less useful than what we are learning is possible. 

I should probably look to other sources for this information I think they presupposed I had some knowledge of convolutional neural networks or at least the layers prior to the course material I am looking at. 

## Pyorch Template for Setting Up Layers. 

dunder init is the place to define your layers there is a helpful page on pytorch's website that lists out all the possible layers you can use. 

forward is the typical function you would use to define the feed forward network behavior it uses those layers you defined in dunder init. there is an image tensor argument to forward where you get your x vector from that would be x. The course provides a sample structure for a class that you'd use for setting up and using a neural network, also it points out that pytorch will do the backprop and also calculate the weight updates for you using autograd. 

here is the example network class file: 

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

def __init__(self, n_classes):
	super(Net, self).__init__()

	# 1 input image channel (grayscale), 32 output channels/feature maps
	# 5x5 square convolution kernel
	self.conv1 = nn.Conv2d(1, 32, 5)
	# maxpool layer
	# pool with kernel_size=2, stride=2
	self.pool = nn.MaxPool2d(2, 2)

	# fully-connected layer
	# 32*4 input size to account for the downsampled image size after pooling
	# num_classes outputs (for n_classes of image data)
	self.fc1 = nn.Linear(32*4, n_classes)
	
# define the feedforward behavior
def forward(self, x):
	# one conv/relu + pool layers
	x = self.pool(F.relu(self.conv1(x)))
	# prep for linear layer by flattening the feature maps into feature vectors
	x = x.view(x.size(0), -1)
	# linear layer 
	x = F.relu(self.fc1(x))
	# final output
	return x

# instantiate and print your Net
n_classes = 20 # example number of classes
net = Net(n_classes)
print(net)
```


ok, and now using that template to do some convolution work, there is a notebook here, don't want to lose it, but let's see if I can learn wha tit is trying to teach me. ok, so we have a class Net which is inheriting from the pytorch nn.Module class. We are using the super constructor at the beginning of our dunder init. Also, that dunder init function is taking weight as a parameter, looks like we slice from the second entry in the weight tensor to the end to get the kernel height and width, that is just giving us the filter size, we then usethe pytorch nn.Conv2d layer initializer to initialze a convolutional layer with 1 dimension for the grayscale channel, then 4 dimensions for I guess the number of filters? then we pass in the kernel size for those filters to be equal to the 2d size of the filters which we got earlier from the weights. we then are setting the weight vector on the conv layer to be equal to torch.nn.Parameter(weight). 

there is also a forward function for the forward propagation step that we already went over in that template, in this case we are just applying that convolutional layer to the x input (that is a parameter to the forward function) and then we are doing an activation step following this to perform ReLU on it. 

So, to use these, lets just use pytorch, from numpy to get the numpy arrays for the filters and create tensors out of them, then we unsqueeze them aby adding a dimension in the 1 dim, which has the effect of adding an array around each of them, that seems to leave us with (4x1x4x4)?? not sure why we added that 1, let me see if I can trace it back through the steps of initializing the net, so I'm still not sure just by looking at the code, because we use the last 2 of the weight dimensions to set up the kernel size, but then the first two are 4 and 1, and that has more to do with the convolutional layer initialization I suppose, lets look at that: 

ok, so the definition of conv2d.weight. it is a tensor that corresponds to the learnable weights of the module and it has the shape (number of out channels, number of in channels/groups, kernel size rows, kernel_size columns) ok, so in this case we are doing 4 out channels, 1 input channel. I guess grayscale is the number of input channels, not sure why we have 4 output channels, I guess there is only one group also, as in only one blocked connection from the input channes to the output channels. Does this mean that we will be creating 4 filters from the one input? Maybe that is why it is 4. come to think of it, that output channels isn't something we had to specify with our unsqueeze it was given by how many filters we had. Ok that makes more sense. 

Ok, then this notebook does something super cool. it creates a viz layer function that takes in a layer and number of filters on it (default 4 becauas that is convenient), and it creates a plotly figure 20px20px in size, then it iterates num filters times and creates axes adding a subplot for each filter, then draws onto it using the numpy squeeze function applied to the current layer in the loop with layer[0,i].data.numpy() and uses the grayscale cmap option adn sets the title to output (i)

actually the viz function looks reaally simple then it is relying on the pytorch layers for the most part, just printing them in a friendly layout. the code that actually plots it is also not that bad. there is separate plotly code to just plot the original car image and then some code to make sub plots for each of the filters that just draws them accoring to what patterns they are identifying in the data. 

Then you have the interesting part, you convert the image into an input tensor for pytorch, all it seems is happening here is you are getting the image data pixels (who knew it was a numpy array already) and then you are unsqueezing once on the 0 dimension and then again on the 1 dimension. so that creates a tensor that is 1x1xwidthxheight I think or maybe heightxwidth idk. guess pytorch needs it to be that shape in order to convolve it, that is a little confusing to me, maybe there is more on the pytorch conv2d page. Ah! yes I can, so the input for conv2d should be of the form (number of examples in tis minibatch, number of channels in the input image, height of the image, width of the image) makes sense that height comes first because matrices want to be (row, column)

so then after we set up the tensor from the grayscale image, we are able to set up the convolution layer both before and after activation using that pytorch model class we created, Net. When you pass the tensor for the input image, looks like the pytorch parent class provides a function that will just execute the forward propagation step directly and return the tuple result that you were returning from forward and you can unpack that into the two results. 

Then, you can visualize them with tha viz function that we wrote earlier, and you can see that the ones that were prior to activation are more gray and lower contrast, but post ReLU activation, the pixels in the filters are high contrast and the patterns that the filter found are sticking out a lot more. 

## Max Pooling Layer

pooling layers take in an image, and they output a reduced version of the image. the dimensionality of the input is reduced as part of this step. So, interestingly, like you convolve over the image in the convolutional layer, you are setting up a size for the pooling layer as well and a stride that can be the entire width of your pooling filter or less I suppose maybe even more, but the point is that you move that pooling filter (is it right to call it a filter? Maybe not) over the image and it figures out based on the maximum pixel value which of the parts of what it is over to keep and throws away the rest. Really interesting, you would think data loss here, but I guess it is not the case. I should probably go back and learn more about this from the coursera specialization as well. 

depending on the size of the area that the pooling layer is looking at, and I suppose the stride size, you can end up with different multipliers of reduced output size. 

Ok, so some intuition is coming. Pooling layers take convolutional layers as input. conv layers are a stack of feature maps where there is a feature map per filter. Depending on how advanced your network is, you might need more and more filters to be able to detect all the different features you want to identify in the image. Doing this doesn't come for free, because the more filters you have, the higher the dimensions you end up having for your network and the more likely it is that you will have your netowrk overfitting to the examples it has already seen. So to prevent this, you use the pooling layers. 

2 different types: 
1. max pooling layer - take a stack of feature maps as the input. there are params, window size and stride. use sliding window technique. just slide the window and take the maximum value of the pixel in the window. 
2. global average pooling - we don't specify window size or stride. just take in the feature maps and compute the average value of the entire feature map for each one in the stack. This is more extreme of a pooling technique. the final output is that you go all the way down from a 3d array and shrink it down to a 1d vector. 

analogy given was a stack of pancakes, you get back the same number of pancakes but they are smaller in max pooling, and then for global pooling you can think of it as getting back a crumb for each feature map pancake that was entered. 

### pooling layers in pytorch

Ok, so the big difference between the example notebook (which I downloaded this time) and the one that came before it is that inside of our Net class in the dunder init we are instantiating a pooling layer nn.MaxPool, and then in the forward function we are doing self.pool(activated_x) after the relu, then returning a tuple of all three layers. 

## Fully Connected Layer + ReLU

the point of these is to connect the input that comes into it into the desired output type, so like a transformation that happens on the backend to ensure that we can epresent things in a way that will be consumable by the user of the network. the transormation that usually occurs here is one tha tconverts the matrix of image features into feature vectors 1 row by C columns where that is the number of possible classes you can have. So for this convolutional neural net, you will typically be transforming into n classes, you would have a set of pooled and activted feature maps as the input and then use a combination of the features, by doing the linear combinations that happen in a typical neural network layer and result in a feature vector that is n items long. the feature maps will be compressed into the feature vector you end up with. 

## Softmax

the last layer is a softmax function which can take any vector of the values you have and return a vector of the same length but all the values in that vector are between 0 and 1 and when you add them all together you get 1. this converts whatever feature vector your fully connected layer ended up with into a probability distribution over which class is most likely. 

these can also be called the class scores for the given model, to extract the most likely class for a given image

## Dropout (optional)

you can add a dropout layer to avoid overfitting. a dropout layer turns off some of the nodes, we saw this in the other training lecture, they will do this uniformly with some probability p. there is an equal chance that any node will be given the opportunity to classify the images for training, and there is a lower chance that only a few heavily weighted nodes will dominate.

## Clothing Classification example

1. so first the dataset is constructed. you get the pytorch dependences and get the fashingmnist dataset which is already an extension of the datasets class. 
2. then, knowing that you will end up with PILImage format in the range [0,1] you cant to create a transformer that will transform this format into a tensor so it can be used like a numpy array that works with GPUs for better performance. You use the ToTensor() method off of torchvision transforms
3. now, if you want to train_data, you instantiate the fashionMNIST dataset which expects a location, whether or not to train the data, whether or not it is the training set of the data, whether or not you are downloading it first and you need to provide that transformer that you prepared so you can store it as tensors. 
4. apparently pytorch has a Dataloader tool that knows how to chop the dataset into batches and shuffle it so let's use that. 
5. we need to specify the names for our classes, so we just pass in an array of the names of the different types of things that we have labeled our training set with. 
6. to visualize the data, matplotlib inline is a ipython magic string that renders the figure of matplotlib in the notebook itself. Then, you get a bartch from the trainloader, there is an iter method that lets you just get one, is that a python thing? does Dataloader return a generator or something? then you get the next item from the ieterator which is your first example from the data, and that is a tuple that contains images and labels so yo ucan unpack that. Following that you need to converty the images (which are tensors iirc) into the numpy array form because matplot and  numpy work well together I guess matplotlib doesn't plot tensors so well. Then you create a simple figure that is 25x4, I'm not sure of what, probably not pixels. then you have the index in the numpy range thta is the same size as the batch size, I guess that is so you get a 1 indexed range. you then add subplots for each item in the batch and you make the subplot capable of plotting each of the images in grayscale on the axis. The way they set up the subplots looks like it is doing something fancy with the 2 rows adn then batch_size/2 columns I guess it knows which index to use as well abecause they add 1 to the index. The results are pretty self explanatory, all the images of that batch. 
7. normalize each of the images. This is a very important step. if you normalize, then during feed forward and backpropagation training in the CNN the image features will fall within a similar range of values and not activate the layers in the network too much. It is like making sure that the features aren't all over the place because then one example could really throw off the weights with differences that are exagerrated but not really a big deal. When normalized, the features are all within a similar range for all examples and that means true differences between the images stand out more and the network will have an easier time converging to an optimal value for the weights because the loss function won't be getting so much noise. I am using hand wavy words like noise and optimal. I think I understand why this is the case intuitively but it wouldn't hurt to do the math on it a few times to really cement it in. 

### from dataset loading to training

you have to define how the model will learn from the data and the core of that is the loss function it uses and athe optimizer. that is how the model will update the parameters and it will determine the speed of convergence. 

Classification usually means you are doing cross entropy loss. there are also some standard stochastic optimizers like stochastic gradient descent and Adam. 

you should also think about whether you are ding regression or classifiction. So this problem will be classifying different articles of clothing, but if you were doing regression to maybe figure out a numerical value like the location in an image for some part of the image, then you would use a different loss function that would help you find that numerical value. 

### training 

you will need to decide how many epochs to use to train the model, through these high level steps: 
1. get all the input images and labeled data together
2. pass the input through the network in a forward pass
3. compute the loss according to the loss function
4. determine the gradient (partial derivative with respect to the weight) for all weights and biases in the model through backpropagation (chain rule)
5. update the weights, the parameter vectors for each of your layers. 

### Ok, actually doing the exercise now, what do I remember? 

Dataloader returns an iterable over the loaded dataset. 
had issues with my softmax and then saw that they were using one from F and it didn't suffer the same dimensionality problems. 

had to look up guides in the pytorch site and read their helpful hints for defining the layers and doing the forward pass implementation. 

I didn't keep track of the dimensions of the output of the convolutional layers, the segment of the course that covered that was not included in the portions of video provided. Their solution repo actually did go into it enough to talk about how to derive the dimensionality of the output with the (w-f)/s + 1 formula. So my performance was bad before applying that knowledge over 5 epochs, not that bad, but to think that I wasn't properly flattening the input to the linear layer probably made a difference. 

so my network wasn't even getting accuracy above zero on a single example when untrained. that's pretty bad. seems like there might be something wrong here because apparently just guessing would give 10% accuracy and this model does worse than that. 

The concept of a Variable Wrapper is just kind of thrown in there for kicks, but it sounds really important. Pytorch uses this to automatically track how the input is changing as it passes through the network and this is how it is able to automatically able to calculate gradients for backprop during a forward pass. Note to self, go into the pytorch underbelly and see how that works sometime. 

Other reasons besides the dimension mixup that my original netowrk was bad:
- I forgot the linear fully connected layer at the end, so I was basically feeding the convolutional layer output directly into a softmax, not really transforming the functional output of the representations I'd learned from the image into anything of note. 

it looks like the code when it was written had the output of max as being a scalar when now it is returning a tensor, so I get a tensor response, but now the accuracy of my model is much better. 

### performance before training

this is I guess a baseline to make sure that the model performs about as well as a random guess to start with. mine was slightly better than that. 
the method here was to go through the images in each mini batch and get the predictions and check them against the labels, then figure out the percentage of the time that we were able to correctly predict the class of the image. 

### actual training

the pytorch framework makes this look so easy. 
1. Looks like we go through the minibatches with an index and data for each of the batches in the train_loader (wehre we loaded the FashionMnist dataset at the beginning) and we get the input images and labels from the data, then assign them to the result of wrappnig them in a variable object. 
2. then we use the zero_grad() option of our optimizer to fill the weights with zeros. I guess that is best practice for convlutional networks, maybe deep neural nets in general? I wonder when it is better to use a stochastic low weight value approach vs. starting with all zeros. I remember something vaguely from coursera about this.
3. do the forward pass. Since Pytorch cleverly must implement the dunder call method under the covers in the parent class, invoking the instance like it was a function is suger for calling this underlying method that probably calls all the internals of the class. It reminds me a lot of like a template method pattern, very elegant indeed. 
4. we get the loss using the criterion that we established above, ours was cross entropy loss, we have to pass the forward pass outputs I guess that is the class scores, as well as our ground truth labels for each of the records in the mini batch, and the criterion takes care of the rest for us
5. then we need to do a backprop step to calculate the gradients for each of the weights this is just taking the loss object we got back from the criterion and calling backward() on it, wonder what type of object that is that makes it possible to do that. 
6. then, this is really strange, we do the optimizer.step which should be updating the weights according to the calculated gradients. Pytorch must be connectin the optimizer and the criterion behind the scenes because there is no argument to be passed into the optimizer for this step, it simply calls step() and then it is a side effect that the params are updated. Ah, looked this up separately, this is part of the way that pytorch works, they opted to store the gradients on the tensor objects themselves, so when you call backwards() on the loss, that updates these gradients on the tensors and then the optimizer has a reference to those from when you instantiated it, so it uses each tensor's own gradient to update its own weights. 
7. loss statistics are prints next, so every 1000 minibatches that you go over it prints out which epoch you are on, what batch you are on and what the loss is now. 

### test accuracy

unfortunately there is a type error somewhere in this 3 year old notebook that prevents me from seeing what my accuracy is. Rather than spending too much time debugging it now, I'll just hope that it was high enough and move on so that I can get back to the DeepQLearning assignment, where I will no doubt have to do a lot of similar troubleshooting but in the context of actual reinforcement learning work with conv nets. 










