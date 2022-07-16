#!/usr/bin/env python
# coding: utf-8

# # Building a chatbot with Deep NLP

# In[1]:


#importing the libraries
import numpy as np
import tensorflow as tf
import re
import time


# In[2]:


##################Part1 Data Preprocessing################

# Importing the Dataset
lines = open('movie_lines.txt' , encoding = 'utf-8', errors= 'ignore').read().split('\n')
conversations = open('movie_conversations.txt' , encoding = 'utf-8', errors= 'ignore').read().split('\n')


# In[3]:


# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) ==5:
        id2line[_line[0]] = _line[4]


# In[4]:


#Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append( _conversation.split(','))


# In[5]:


# Getting separately the questions and the answers 
questions =[]
answers=[]
for conversation in conversations_ids:
    for i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# In[6]:


# Doing the first cleaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm" , "i am", text)
    text =re.sub(r" he's", " he is", text)
    text = re.sub(r"she's", "she is", text)
    text= re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text=re.sub(r"where's","where is", text)
    text=re.sub(r"\'ll", " will", text)
    text= re.sub(r"\'ve"," have", text)
    text=re.sub(r"\'re", " are", text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't"," will not", text)
    text=re.sub(r"can't"," cannot",text)
    text=re.sub(r"[-()\"# /@;:<>{}+=~|. ? , ]","",text)
    return text


# In[7]:


#Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))


# In[8]:


#Cleaning the answers
clean_answers= []
for answer in answers:
    clean_answers.append(clean_text(answer))
    


# In[9]:


#Creating the dictionary that maps each word to its number of occurances
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            


# In[10]:


# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshhold = 20
questionswords2int = {}
word_number =0
for word,count in word2count.items():
    if count >= threshhold:
        questionswords2int[word] = word_number
        word_number += 1 
answerswords2int = {}
word_number =0
for word,count in word2count.items():
    if count >= threshhold:
        answerswords2int[word] = word_number
        word_number += 1        
        


# In[11]:


#Adding the last tokens to these dictionaries
tokens =['<PAD>','<EOS>','<OUT>','<EOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1    


# In[12]:


# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word={w_i: w for w,w_i in answerswords2int.items()}


# In[13]:


# Adding the End of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += '<EOS>'


# In[14]:


# Translating all the questions and answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints =[]
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)        


# In[15]:


#Sorting questions and answers by the length of questions
#resason 4 this : cause it speeds up training & reduce the Loss
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):
    for i in enumerate(questions_into_int):
        if len(i [1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


# In[16]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf2 #Tensorflow 2.x


# In[17]:


############################################## PART2 #######################################
#Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name= 'input')
    targets = tf.placeholder(tf.int32, [None,None], name= 'input')
    lr = tf.placeholder(tf.float32, [None,None], name= 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


# In[18]:


# Preprocessing the targets
def preprocess_target(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    return preprocessed_targets    


# In[19]:


#Creating the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequance_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequance_length=sequance_length,
                                                       inputs =rnn_inputs,
                                                       dtype=tf.float32
    )
    return encoder_state


# In[20]:


#Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embeded_input, sequence_length, decoding_scope,output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_sell])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option = "bahdanau",num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             name="attn_dec_train")
    decoder_output,decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            training_decoder_function,
                                                                                                            decoder_embedded_input,
                                                                                                            sequance_length,
                                                                                                            scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# In[21]:


#Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeded_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope,output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_sell])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option = "bahdanau",num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                             encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                              decoder_embedding_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name="attn_dec_inf")
    test_predictions,decoder_output,decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                              test_decoder_function,
                                                                                                                              scope = decoding_scope)
    return test_predictions


# In[22]:


#Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequance_length, rnn_size, num_layers, word2int, keep_pro,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializers = weights,
                                                                      biases_initializers = biases
        )
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequance_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size
        )
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                          decoder_sell,
                                          decoder_embeddings_matrix,
                                          word2int['<SOS>'],
                                          word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions,test_predictions
                                           


# In[23]:


#Building seq2seq model
def seq2seq_model(inputs,targets, keep_prob, batch_size, sequence_len, answers_num_words, question_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int ):
    encoder_embedded_input = tf.compat.v1.estimator.layers.embed_sequence(inputs,
                                                             answers_num_words +1,
                                                             encoder_embedding_size,
                                                             initializer = tf.random_uniform_initializer(0,1)
                                                              # initializer= tf2.initializers.GlorotUniform()
                                                             )
    encoder_state = encoder_rnn(encoder_embedded_input,rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_mtrix = tf.Variable(tf.random_uniform([questions_num_words +1, decoder_embedding_size],0,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embedding_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


# In[24]:


################################PART3- TRAINING the seq2seq Model ###########################################
#Setting the Hyperparameters
epochs=100
batch_size =64
rnn_size = 512
num_layers =3
encoding_embedding_size =512
decoding_embedding_size = 512
learning_rate =0.01
learning_rate_decay=0.9
min_learning_rate = 0.0001
keep_probability = 0.5


# In[25]:


# #Defining a session
# tf.reset_default_graph()
# session = tf.InteractiveSession()


# In[26]:


# #Defining a session
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# session = tf.InteractiveSession()


# In[27]:


#Defining a session
from tensorflow.python.framework import ops
ops.reset_default_graph()
session =tf.compat.v1.InteractiveSession()


# In[28]:


#Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()


# In[29]:


#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length') 


# In[30]:


# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)


# In[31]:


import tensorflow as tf

print(tf.__version__)


# In[34]:


from tensorflow_estimator.python.estimator import layers


# In[32]:


# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      batch_size,
                                                      sequence_length,
                                                      len(answerswords2int),
                                                      len(questionswords2int),
                                                      encoding_embedding_size,
                                                      decoding_embedding_size,
                                                      rnn_size,
                                                      num_layers,
                                                      questionswords2int)


# In[ ]:


#Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error =tf.compat.v1.estimator.seq2seq.sequence_loss(training_predictions,
                                                 targets,
                                                 tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable) for grad_tensor, gard_variables in grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# In[ ]:


#Padding the sequences with the <PAD> token
#Question: [ 'who' ,'are', 'you', '<PAD>','<PAD>','<PAD>','<PAD>']
#Answer : [ <SOS>, "I" ,'am', 'a', 'bot' ,'.' ,'<EOS>', '<PAD>']
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequences in batch_of_sequences])
    return [sequence + [wordint['<PAD>']]*(max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# In[ ]:


#Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)// batch_size):
        start_index = batch_index * batch_size
        questions_in-batch = questions[start_index : start_index + batch_index]
        answers_in_batch = answers[start_index : start_index + batch_index]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch,answerswords2int))
        yield padded_questions_in_batch , padded_answers_in_batch


# In[ ]:


#Splitting the questions and answers into training and validation sets
training_validation_split =int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers=sorted_clean__answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


# In[ ]:


#Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer)
for epoch in range(1,epochs + 1):
    for batch_index, (padded_questions_in_batch,padded_answers_in_batch) in  enumerate(split_into_batches(training_questions,training_answers,batch_size)):
    starting_time =time.time()
    _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],{inputs:padded_questions_in_batch,
                                                                                         targets:padded_answers_in_batch,
                                                                                         lr:learning_rate,
                                                                                         sequnce_length:padded_answers_in_batch.shape[1],
                                                                                         keep_prob:keep_probability})
    total_training_loss_error += batch_training_loss_error
    ending_time=time.time()
    batch_time = ending_time - starting_time
    if batch_index % batch_index_check_training_loss == 0:
        print('Epoch: {:>3}/{}, Batch :{:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches:{:d} seconds'.format(epoch,
                                                                                                                                epochs,
                                                                                                                                 batch_index,
                                                                                                                                 len(training_question)//batch_size,
                                                                                                                                 total_training_loss_error/
                                                                                                                                 batch_index_check_training_loss,
                                                                                                                                 int(batch_time * batch_index_check_training_loss)))
        
        total_training_loss_error=0
        if batch_index % batch_index_check_training_loss == 0 and batch_index >0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation , (padded_questions_in_batch, padded_answers_in_batch) in  enumerate(split_into_batches(validation_questions, validation_answers))
                batch_validation_loss_error = session.run(loss_error,(inputs: padded_questions_in_batch)
                                                         targets:padded_answers_in_batch
                                                         lr:learning_rate,
                                                         sequence_length:padded_answers_in_batch.shape[1],
                                                         keep_prob:1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time=ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds' .format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now !!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session,checkpoint)
            else:
                print('Sorry I do not speak better, I need to practice more.')
                early_stopping_check +=1
                if early_stopping_check == early_stopping_stop:
                    break
            if eraly_stopping_check == early_stopping_stop:
                print('My appologies, I cannot speak better anymore. This is the best Ican do.')
                break
            print('Game over')    


# In[ ]:


###########################PART4 TESTING SEQ2SEQ MODEL############################


#Loading the weights and Running the session
checkpoint = "./ chatbot_weights.ckpt"
session=tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver=tf.train.Saver()
saver.restore(session, checkpoint)


# In[ ]:


#Converting the questionsfrom strings to lists of encoding integers
def convert_string2int(question,word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


# In[ ]:


#Setting up the chat
while(True):
    question = input("You: ")
    if question == "Goodby":
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (20 - len(question))
    fake_batch = np.zeros((batch_size, 20))
    fake_batch [0] = question
    predicted_answer = session.run(test_predictions, {inputs : fake_batch , keep_prob : 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer,1):
        if answersints2word[i]=='i':
            token =' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersintsword[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
        print('ChatBot: ' + answer)    


# In[ ]:


# questionswords2int


# In[ ]:


# questions


# In[ ]:


#id2line


# In[ ]:


#lines


# In[ ]:





# In[ ]:




