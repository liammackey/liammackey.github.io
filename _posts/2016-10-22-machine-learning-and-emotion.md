---
title:  "Machine Learning, Emotion, and Python"
date:   2016-10-22 20:46:23
categories: [machine learning]
tags: [machine learning]
---

Today embarked on my first Machine Learning Project. It was simple and could, if need be, be condensed down to just nine lines of python. The idea here was inspired by a blog post that gave instructions on how to build your very own neural network from scratch. Although mine wasn’t as much a network as it was just one Neuron. The code I wrote basically came down to just performing some Matrix multiplication on a few input nodes with a little back propagation thrown in there. I am still in the process of learning how to create Machines that think but the idea of the thinking machine has excited me since I was a young boy. 

Participating in my high school's robotics team definitely provoked me to begin thinking about technology and where we will be in 20 years time in terms of machines and Artificial Intelligence. The concept of a machine that can think has been around for a long time but not until recently have we begun to make significant strides in making it happen. Tesla just announced that it will be deploying all of its new vehicles with fully autonomous driving capability. This was a genius move by their engineers. The thing that powers all Neural Networks is data. And by making all new cars data collection machines Tesla can gather in a few weeks the amount of data that it took years for google cars to gather, thus making their Artificial Intelligence (AI) much more powerful. 

Another recent milestone in Natural Language Processing and Machine Learning was Microsoft achieving parity between their Machine’s Speech Recognition and the average human speech recognition. This was a major accomplishment. This means that Microsoft’s AI now detects human speech probably as good as you do. 

Although AI is an enticing idea, we as a species need to proceed with caution. Some of the world's smartest engineers and scientists have warned of a future where Artificially Intelligent Robots wipe out humanity. The Terminator is an actual possibility in many Scientists and Engineers minds. We as the Engineers who create the machines must always make sure there is a Kill switch. Just like when Photoshop takes over you Mac and you can't shut it down, we need to make sure we have the ability to `sudo kill PID`. 



The idea of sentient Machines has always been appealing to me. The human consciousness is so vast and filled with high peaks and low valleys that to be able to replicate that in a machine would be a huge feat. However it brings up an interesting question, If we could replicate consciousness would we mimic it after our own or make it better. Would we want to give our Robots and AI’s the same flaws that we as humans deal with? The undue Anxiety and Sadness that we cause ourselves? If we did that it might make them more easy to relate too, but it would also create a factor of unpredictability and maybe even suffering in the machines we create. If we didn't imbue on them all the peculiarities of human suffering would they even be conscious? Isn’t it our emotions that bring life to our consciousness. We view the world through the lens of our feelings, and if we don't give our machines the whole array of human emotion will they ever be able to be truly conscious in our eyes? Because as of now the only intelligent and conscious beings we have ever met are our fellow humans, so what would it be like to be a conscious without all the emotions that make our consciousness worthwhile? The technology to build a truly conscious machine is still far off (maybe even impossible) my program is only able to predict simple number patterns, but better to start thinking about these issues now rather then when it's too late. 

Either way Machine Learning is becoming an integral part of our everyday lives from the Taco Bell Slack Chatbot to Tesla’s fully Autonomous cars we are entering the age of the machines. It is coming and we can't stop it so best we proceed with caution. On the other hand, I for one Welcome our new Robot Overlords!


![alt text](/images/Robot.jpg)


The Python code for anyone interested:

```python

from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))

```