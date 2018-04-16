import numpy as np
from faker import Faker
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt



def load_dataset(m):
    
    '''generates invoice data set 
    Arguments:
    m -- total number of dataset for invoice number
    Returns:
    dataset -- X,Y values for training
    by_ocr --  a python dictionary mapping all characters used in the ocr readable invoice number to an integer-valued index
    machine -- a python dictionary mapping all characters of expected invoice number to an integer-valued index.
    inv_machine -- the inverse dictionary of machine_vocab, mapping from indices back to characters.
    '''
    distribution=m//4
    training_set_x=[]
    training_set_y=[]
    label_y_str="Invoice Number: "
    ocr_vocab=set()
    machine_vocab=set()
    fake=Faker()
    inv_strings=['Commercial invoice number: ','Commercial invoice num: ', 'Commercial inv no: ','Commercial invoice #: ', 'Commercial invoice abc: ', 'Commercial invoice ahdoug: ', "Commercial invoice kagkaef: ", "Commercial invoic no.: "]



    for i in inv_strings:
        for k in range(distribution):
            tmp=str(fake.random_number(digits=4,fix_len=True))+"-"+str(fake.random_number(digits=4,fix_len=True))+"-"+str(fake.random_number(digits=4,fix_len=True))
            x=i+tmp
            y=label_y_str+tmp
            ocr_vocab.update(tuple(x))
            machine_vocab.update(tuple(y))
            training_set_x.append(x)
            training_set_y.append(y)
    for i in inv_strings:
        for k in range(distribution):
            tmp="IN"+str(fake.random_number(digits=5,fix_len=True))
            x=i+tmp
            y=label_y_str+tmp
            ocr_vocab.update(tuple(x))
            machine_vocab.update(tuple(y))
            training_set_x.append(x)
            training_set_y.append(y)
    for i in inv_strings:
        for k in range(distribution):
            tmp=str(fake.random_number(digits=5, fix_len=True))
            x=i+tmp
            y = label_y_str + tmp
            ocr_vocab.update(tuple(x))
            machine_vocab.update(tuple(y))
            training_set_x.append(x)
            training_set_y.append(y)
    for i in inv_strings:
        for k in range(distribution):
            tmp=str(fake.random_number())
            x=i+tmp
            y = label_y_str + tmp
            ocr_vocab.update(tuple(x))
            machine_vocab.update(tuple(y))
            training_set_x.append(x)
            training_set_y.append(y)

    dataset=list(zip(training_set_x,training_set_y))
    by_ocr = dict(zip(sorted(ocr_vocab) + ['<unk>', '<pad>'],
                     list(range(len(ocr_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    inv_machine.update({len(inv_machine):None})
    machine = {v: k for k, v in inv_machine.items()}
    return dataset,by_ocr,machine,inv_machine
	
def preprocess_data(dataset, ocr_vocab, machine_vocab, Tx,Ty):
    '''one-hot versions'''
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, ocr_vocab,"x") for i in X])
    
    Y = [string_to_int(t, Ty, machine_vocab,"y") for t in Y]
    #print("Yyyyy",Y)

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(ocr_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh
	
def string_to_int(string, length, vocab, var):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    #rep=[]
    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    if var == "x":
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(string))
    else:
        if len(string) < length:
            rep += [vocab[None]] * (length - len(string))
    return rep

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')