# Inspired from :https://datascience.stackexchange.com/questions/94205/a-simple-attention-based-text-prediction-model-from-scratch-using-pytorch

# authur: me__unpredictable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from self_attntion_model import Model
# random.seed(0)
# torch.manual_seed(0)

# Sample text for Training
test_sentence = """The sayings of King Lemuel—an inspired utterance his mother taught him. Listen, my son! Listen, son of my womb! Listen, my son, the answer to my prayers! Do not spend your strength[a] on women, your vigor on those who ruin kings. It is not for kings, Lemuel—
    it is not for kings to drink wine, not for rulers to crave beer, lest they drink and forget what has been decreed, and deprive all the oppressed of their rights. Let beer be for those who are perishing,
    wine for those who are in anguish! Let them drink and forget their poverty and remember their misery no more. Speak up for those who cannot speak for themselves,
    for the rights of all who are destitute. Speak up and judge fairly; defend the rights of the poor and needy. A wife of noble character who can find?
    She is worth far more than rubies. Her husband has full confidence in her and lacks nothing of value. She brings him good, not harm,
    all the days of her life. She selects wool and flax and works with eager hands. She is like the merchant ships,
    bringing her food from afar. She gets up while it is still night; she provides food for her family and portions for her female servants. She considers a field and buys it;  out of her earnings she plants a vineyard. She sets about her work vigorously; her arms are strong for her tasks. She sees that her trading is profitable,
    and her lamp does not go out at night. In her hand she holds the distaff and grasps the spindle with her fingers. She opens her arms to the poor
    and extends her hands to the needy. When it snows, she has no fear for her household; for all of them are clothed in scarlet. She makes coverings for her bed;
    she is clothed in fine linen and purple. Her husband is respected at the city gate, where he takes his seat among the elders of the land. She makes linen garments and sells them,
    and supplies the merchants with sashes. She is clothed with strength and dignity; she can laugh at the days to come. She speaks with wisdom,
    and faithful instruction is on her tongue. She watches over the affairs of her household and does not eat the bread of idleness. Her children arise and call her blessed;
    her husband also, and he praises her: “Many women do noble things, but you surpass them all.” Charm is deceptive, and beauty is fleeting;
    but a woman who fears the Lord is to be praised. Honor her for all that her hands have done, and let her works bring her praise at the city gate.""".lower().split()

# Build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
prediction_set = [([test_sentence[i], test_sentence[i + 1]], [test_sentence[i + 2],test_sentence[i + 3],test_sentence[i + 4],test_sentence[i + 5]])
            for i in range(len(test_sentence) - 5)]

# print the first 3, just so you can see what they look like
# print(prediction_set[:3])

vocab = list(set(test_sentence))
print(vocab)
# here we create key using words, their location/indexes are their value
# i.e. 'who':0,'hello':1
# this helps to find location of word in vocab
# these values will be used to predict
key_value_map = {word: i for i, word in enumerate(vocab)}

# ------------------------------------------------------------------------------
# Model parameters
# output size
# here in  trigrams we have 4 tensors in output hence output size is 4
OUTPUT_SIZE=4

# Number of Epochs
EPOCHS = 10

# SEQ_SIZE is the number of words we are using as a context for the next word we want to predict
SEQ_SIZE = 4

# Embedding dimension is the size of the embedding vector
EMBEDDING_DIM = 10

# Size of the hidden layer
HIDDEN_DIM = 256

learning_rate = 0.001
loss_function = nn.NLLLoss()  # negative log likelihood
CONTEXT_SIZE=2
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Model initialization
model = Model(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM,OUTPUT_SIZE)
model.to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
# Model training
loss_=[]
for i in range(EPOCHS):
    total_loss = []
    for context, target in prediction_set:
        # context, target = ['thomas', 'edison.'] the
        # step 1: context id generation
        context_idxs = torch.tensor([key_value_map[w] for w in context], dtype=torch.long)
        context_idxs=context_idxs.to('cuda')
        # step 2: setting zero gradient for models
        model.zero_grad()

        # step 3: Forward propogation for calculating log probs
        log_probs = model(context_idxs)
        # print('Output shape:',log_probs.shape)
        # step 4: calculating loss
        target=torch.tensor([key_value_map[w] for w in target])
        target=target.to('cuda')
        loss = loss_function(log_probs,target)

        # step 5: finding the gradients
        loss.backward()

        # step 6: updating the weights
        optimizer.step()

        total_loss.append(loss.item())
    total_loss=np.mean(total_loss)
    print('\r',end=' ')
    print("Epoch: ", str(i), " Loss: ", str(total_loss),end=' ')
    loss_.append(total_loss)
# -------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Loss plot
plt.plot(loss_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend(['Loss'])
plt.grid()
plt.show()
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Model inference
with torch.no_grad():
    # Fetching a random context and target
    rand_val = prediction_set[random.randrange(len(prediction_set))]
    print(rand_val)
    context = rand_val[0]
    target = rand_val[1]

    # Getting context and target index's
    context_idxs = torch.tensor([key_value_map[w] for w in context], dtype=torch.long)
    context_idxs=context_idxs.to('cuda')
    target_idxs = torch.tensor([key_value_map[w] for w in target], dtype=torch.long)
    target_idxs=target_idxs.to('cuda')
    print("Input : ", context_idxs.to('cpu').detach().tolist(),'Expected Prediction:' ,target_idxs.to('cpu').detach().tolist())
    log_preds = model(context_idxs)
    prediction=torch.argmax(log_preds,dim=1).cpu().detach()
    print('RAW Prediction:', prediction)
    # print("Predicted indices: ", torch.argmax(log_preds),vocab[torch.argmax(log_preds[0]).cpu().detach()])
    print('This is how it looks like:\n',vocab[context_idxs[0]],vocab[context_idxs[1]],vocab[prediction[0]],vocab[prediction[1]],vocab[prediction[2]],vocab[prediction[3]])
# --------------------------------------------------------------------------------
