import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, SGD
from keras import regularizers
from logger import Logger
from data_utils import *

INPUT_SHAPE = (27,27,3)

def get_arguments():
	parser = {}
	parser['basepath'] = "../CrackForest-dataset"
	parser['pretrained'] = 0
	parser['optimizer'] = 'rms'
	parser['modelfile'] = "my-model.h5"
	parser['lr'] = 1e-4
	parser['savedir'] = "parameters"
	parser['epoch'] = 5
	parser['saveprediction'] = "output"
	return parser

def seg_model(n_classes):
	model = Sequential()
	model.add(Conv2D(16, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),
		input_shape=INPUT_SHAPE))
	model.add(Conv2D(16, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(32, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Conv2D(32, (3,3),
		strides=(1, 1), padding='same',
		use_bias=True, activation='relu',
		kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#Fully connected layers
	model.add(Flatten())
	model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(25, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

	return model


if __name__ == '__main__':
	args = get_arguments()
	base_path = args["basepath"]
	image_path = os.path.join(base_path, 'image')
	gt_path = os.path.join(base_path, 'groundTruth')
	gt_files = os.listdir(gt_path)
	train_gt, val_gt = validation_split(gt_files, 0.2)
	print("training on {} samples, validating on {} samples".format(len(train_gt), len(val_gt))) 

	save_dir = args["savedir"]
	epoch = args["epoch"]
	batch_obj = Logger('batch','batch.log','info')
	logger_batch = batch_obj.log()

	if args["pretrained"] == 0:
		model = seg_model(2)
		if args["optimizer"] == 'rms':
			optimizer = RMSprop(lr=args["lr"])
		elif args["optimizer"] == 'sgd':
			optimizer = SGD(lr=args["lr"], decay=1e-6, momentum=0.9, nesterov=False)
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	else:
		model = load_model(args["modelfile"])

	epoch_count = 0
	while epoch_count < epoch:
		#perform training
		batch_count = 0
		train_gen_object = load_gen_v2(image_path, gt_path, train_gt)#, args.batchsize, 2)
		for batch_train in train_gen_object:
			x,y = batch_train
			try:
				assert(x.shape[0] == y.shape[0])
				loss, accuracy = model.train_on_batch(x,y)
				logger_batch.info('training loss for epoch_no {} batch_number {} is loss:{}, accuracy:{}'.format(epoch_count, batch_count, loss, accuracy))
				batch_count+=1

			except Exception as e:
				print(e)
				continue

		#perform validation
		batch_count,total_loss = 0,0
		val_gen_object = load_gen_v2(image_path, gt_path, val_gt)#, args.batchsize, 2)
		for batch_val in val_gen_object:
			x,y = batch_val
			try:
				assert(x.shape[0] == y.shape[0])
				loss, accuracy = model.test_on_batch(x,y)
				batch_count+=1
				total_loss+=loss
			except Exception as e:
				print(e)
				continue

		logger_batch.info('validation loss for epoch_no {} is loss:{}, accuracy:{}'.format(epoch_count, (total_loss/batch_count), accuracy))

		filename = 'mymodel'+str(epoch_count)+'.h5'
		model.save(os.path.join(save_dir, filename))
		epoch_count+=1