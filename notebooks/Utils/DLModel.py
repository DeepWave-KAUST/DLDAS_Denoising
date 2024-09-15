'''
	Guided DL for DAS Denoising
	Dec, 10, 2023
	by Omar M. Saad
'''   
# Import Libs.
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Lambda, concatenate, Flatten, Add, Reshape, Dense, Input, GlobalAvgPool1D, Activation, multiply
from Utils.Utils import *
from keras.layers import Lambda
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K


# Attention Network.
def Attention(inp1):
    filters = inp1.shape[-1] 
    sa   = (1,filters)
    inp2 = GlobalAvgPool1D()(inp1)
    inp2 = Reshape(sa)(inp2)

    x = Dense(filters, activation='relu')(inp2)
    x = Dense(filters, activation='sigmoid')(x)
    x = multiply([inp1,x])
    x = Add()([x,inp1])

    return x

# Core Block of the Compact Layer.
def Block(inp,D): 
    
    # Fully-connected Layers.
    x = Dense(D,activation='relu')(inp)
    x = Dense(D,activation='relu')(x)
    x = Reshape((x.shape[-1],1))(x)

    return x

# Compact Layer which extracts the spatial relation between the CWT scale and band-pass filtered data.
def compactlayer(y,D):
    
    # Split the Data into Two Channels (Branches).
    s0 = Lambda(lambda x:x[:,:,0])(y)
    s1 = Lambda(lambda x:x[:,:,1])(y)
    
    # The Band-pass Filter Branch
    B1 = Block(s0,D)
    # The CWT Scale Branch.
    B2 = Block(s1,D)
    # Applying the Attention to the Concatenated Data.
    B    = concatenate([B1,B2], axis=-1)
    Batt = Attention(B)

    return Batt


# Training the DL Model.
def Train(CWT,BP,EPOCHNO,BATCHSIZE,w1,w2,s1z,s2z,D1,eq):
    
    # Normalize the CWT Scale
    ma = np.max(np.abs(CWT))
    dataInput = CWT/ma

    #ma1 = np.max(np.abs(dataInputF))
    #dataInputF = dataInputF/ma1

    # Patching the CWT SCALE.
    dataInputP = patch(dataInput,w1,w2,s1z,s2z)   
    dataInput2 = np.reshape(dataInputP,(dataInputP.shape[0],w1*w2,1))
    
    # Patching the Band-pass Filtered Data.
    dataInputF = patch(BP,w1,w2,s1z,s2z)   
    dataInputF2 = np.reshape(dataInputF,(dataInputF.shape[0],w1*w2,1))

    
    # Setting the Inputs to the DL Model.
    input_shape = (w1, w1,1)
    inp1 = layers.Input(shape=(w1*w2,1),name='input_layer1')
    inp2 = layers.Input(shape=(w1*w2,1),name='input_layer2')

    # Setting the Number of Neurons for the Encoder/Decoder.
    D2 = int(D1/2)
    D3 = int(D2/2)
    D4 = int(D3/2)
    D5 = int(D4/2)
    D6 = int(D5/2)

    # Concatenate the Patches.
    inp4 = concatenate([inp1,inp2])

    # Encoder Compact Layers
    e1 = compactlayer(inp4,D1)
    e2 = compactlayer(e1,D2)
    e3 = compactlayer(e2,D3)
    
    # Decoder Compact Layers
    d4 = compactlayer(e3,D2)
    d5 = compactlayer(d4,D1)
    d6 = compactlayer(d5,D1)

    # Output Layer
    d6 = Flatten()(d6)
    y = Dense(w1*w2, activation='linear')(d6)

    # Creating the Model
    model = Model(inputs=[inp1,inp2], outputs=[y])    

    # Schedule the Learning Rate.
    A = 50
    def lr_schedule(epoch):
        initial_lr = 1e-3

        if epoch <= A:
            lr = initial_lr
        elif epoch <= A+10:
            lr = initial_lr / 2
        elif epoch <= A+30:
            lr = initial_lr / 10
        else:
            lr = initial_lr / 20
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=0.5,
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6,
                                       monitor='loss')

    # Setting the Optimizer and the Loss Function. Also, Saving the Best Model Based on the Loss.
    sgd = tensorflow.keras.optimizers.Adam()
    model.compile(optimizer=sgd, loss=tensorflow.keras.losses.LogCosh(),metrics=['mse'])
    early_stopping_monitor = EarlyStopping(monitor= 'loss', patience = 5)
    checkpoint = ModelCheckpoint('./models/DAS_PATCH_N_eq_' + str(eq) + '.h5',
                                 monitor='loss',
                                 mode = 'auto',
                                 verbose=0,
                                 save_weights_only=True,
                                 save_best_only=True)

    callbacks0 = [lr_scheduler]
    callbacks = [lr_scheduler, early_stopping_monitor, checkpoint]

    # Print the Model Summary.
    model.summary()
    # Training the DL Model
    history = model.fit([dataInputF2,dataInput2], [dataInputF], epochs=EPOCHNO, batch_size=BATCHSIZE, shuffle=True, callbacks = callbacks, verbose = 1)

    # Loading the optimal Model.
    model.load_weights('./models/DAS_PATCH_N_eq_' + str(eq) + '.h5')
    out = model.predict([dataInputF2,dataInput2],batch_size=32)
    # Just in Case We Normalize the Band-pass Filtered Data.
    #out = out*ma1
    # Unpatching to Reconstruct the Input Data.
    out = np.reshape(out,(out.shape[0],w1*w2))     
    outA = np.transpose(out)
    n1,n2=np.shape(BP)
    outB = patch_inv(outA,n1,n2,w1,w2,s1z,s2z)
    outB = np.array(outB)

    # Saving the Denoised Data.
    #np.save(r'denoised_128_eq_' + str(eq),outB)
    # Release the GPU Memory.
    K.clear_session()
    
    return outB