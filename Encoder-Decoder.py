import os 
import pickle
import numpy as np
import collections
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout, concatenate, Flatten
from keras.models import Sequential
import copy
from keras.utils import to_categorical
from keras.models import Model
from keras import optimizers
from keras.utils.vis_utils import plot_model


characterLength=32
vocab_size=100



def build_dataset(Sentences):


        counter = collections.Counter()
        for Sent in Sentences:
            for Character in Sent:
                counter[Character] += 1
        counter_len = len(counter)

        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common(vocab_size - 1))}


        assert "UNK" not in dictionary

        dictionary.update({k:v+1 for k, v in dictionary.items()})
        dictionary["<PAD>"] = 0
        dictionary["UNK"] = len(dictionary)
        dictionary["<s>"]=len(dictionary)



        inverted_dictionary = {value: key for key, value in dictionary.items()}



        data = []
        for contex in Sentences:
            paragraph = []
            for i in contex:
                id_ = dictionary[i] if i in dictionary else dictionary["UNK"]
                paragraph.append(id_)
            data.append(paragraph)

        return data, dictionary, inverted_dictionary




def LoadData():

    DataLoaded=False
    try:
        with open("Sentences.pickle", 'rb') as f:
            Sentences = pickle.load(f)
        DataLoaded=True

    except Exception as e:
        print(e)


    if(not  DataLoaded):
        all_files = os.listdir("data/")
        print(all_files)



        Sentences=[]
        for i in range(1):
            f = open("data/"+all_files[i], "r")
            line = f.readline()
            #print(line)
            #exit()
            while line:

                 #print(line)
                 CharacterBreak=[]
                 for i in range(len(line)):
                    if(i<characterLength):
                        CharacterBreak.append(line[i])
                    else:
                        break
                 CharacterBreak=[x.lower() for x in CharacterBreak]
                 Sentences.append(CharacterBreak)





                 line = f.readline()


        pickle.dump(Sentences, open("Sentences.pickle", "wb"))


    return Sentences



def Train(data,dictionary):

    vocab_size=len(dictionary)

    encoder_input_data=pad_sequences(maxlen=characterLength, sequences=data, padding="post")



    decoder_input_data = copy.deepcopy(encoder_input_data)
    decoder_input_data=decoder_input_data.tolist()
    for i in range(len(decoder_input_data)):
        decoder_input_data[i].insert(0,dictionary['<s>'])
    decoder_input_data=np.asarray(decoder_input_data)



    decoder_target_data = copy.deepcopy(decoder_input_data)
    decoder_target_data=decoder_target_data.tolist()
    for i in range(len(decoder_target_data)):
        decoder_target_data[i].pop(0)
        decoder_target_data[i].append(0)





    #
    #

    #print(np.asarray(encoder_input_data).shape)
    #encoder_input_data = to_categorical(encoder_input_data)
    #decoder_input_data = to_categorical(decoder_input_data)
    encoder_input_data=np.reshape(encoder_input_data,(encoder_input_data.shape[0],encoder_input_data.shape[1],1))
    decoder_input_data=np.reshape(decoder_input_data,(decoder_input_data.shape[0],decoder_input_data.shape[1],1))
    decoder_target_data = to_categorical(decoder_target_data,num_classes=len(dictionary))


    print(decoder_input_data.shape)
    print(decoder_target_data.shape)
    print(encoder_input_data.shape)

    ####################################

    encoder_inputs = Input(shape=(None, 1))
    encoder = LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoderoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, 1))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(len(dictionary), activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training

    optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.9, amsgrad=False)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  
                  metrics=['accuracy'])


    model.load_weights('EncoderDecoder.h5')
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #           batch_size=256,
    #           epochs=5,
    #           validation_split=0.2)

    # model.save('EncoderDecoder.h5')

    return encoder_input_data


def predict(dictionary,inverted_dictionary,input_seq):


    output=[]





    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = dictionary["<s>"]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)


   
        output.append(output_tokens)


        # Sample a token


        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = inverted_dictionary[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (len(output) > max_decoder_seq_length):
            stop_condition = True


        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    output=np.asarray(output)

    output=np.reshape(output,(output.shape[0],output.shape[3]))
    #print(output.shape)


    # for i in output:
    #     #print(i)
    #     print(inverted_dictionary[np.argmax(i)])


    return output
    #return decoded_sentence




############# TRAINING AND MAKING THE Encoder-Decoder to produce soft continous outputs to improve onehot encoding for Generator 
Sentences = LoadData()


data,dictionary,inverted_dictionary=build_dataset(Sentences)
pickle.dump(dictionary, open("dictionary.pickle", "wb"))
pickle.dump(inverted_dictionary, open("inverted_dictionary.pickle", "wb"))


encoder_input_data=Train(data,dictionary)



######################### MAKING AND LOADING MODEL FOR PREDICTING #############################################
max_decoder_seq_length=characterLength
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(dictionary), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.load_weights('EncoderDecoder.h5')

###
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


############################# TIME TO MAKE PREDICTIONS AND STORING AS PICKLE ###################



outputs=[]
#len(encoder_input_data)

print(len(encoder_input_data))
i=int(len(encoder_input_data)/2)
while (i<len(encoder_input_data)):
    print(i)
    outputs.append(predict(dictionary,inverted_dictionary,encoder_input_data[i:i+1]))
    i=i+1


outputs=np.asarray(outputs)
print(outputs.shape)
pickle.dump(outputs, open("SoftOutputs.pickle", "wb"))









##################3





    
