from keras.datasets import mnist
import numpy as np
np.random.seed(10)

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec

from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn import metrics
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose

from encoder_function import autoencoder, autoencoderConv2D_1, autoencoderConv2D_2, ClusteringLayer, target_distribution

def Basic_Kmeans(images, labels, n_init, n_jobs=-1):
	x = images.reshape((images.shape[0], -1))
	y = labels
	n_clusters = len(np.unique(y))
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
	y_pred = kmeans.fit_predict(x)
	nmi = metrics.mutual_info_score(y,y_pred)
	ari = metrics.adjusted_rand_score(y,y_pred)
	print('nmi = %.5f, ari = %.5f' % (nmi, ari))
	return nmi, ari

def AutoEncoder_Kmeans(images,labels, n_init, n_jobs=-1,
		save_dir='./results/screen_clustering/',
		learning_rate=0.01,
		momentum=0.9,
		pretrain_epochs=600,
		batch_size=200):
	x = images.reshape((images.shape[0], -1))
	y = labels
	n_clusters = len(np.unique(y))

	#pretrain autoencoder
	dims = [x.shape[-1], 500, 500, 2000, 6]
	init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform')
	pretrain_optimizer = SGD(lr=learning_rate, momentum=momentum)
	autoencoder1, encoder1 = autoencoder(dims, init=init)
	autoencoder1.compile(optimizer=pretrain_optimizer, loss='mse')
	autoencoder1.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
	autoencoder1.save_weights(save_dir + '/ae_weights.h5')
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
	y_pred = kmeans.fit_predict(encoder1.predict(x))
	nmi = metrics.mutual_info_score(y,y_pred)
	ari = metrics.adjusted_rand_score(y,y_pred)
	print('nmi = %.5f, ari = %.5f' % (nmi, ari))
	return nmi, ari

def DEC_Kmeans(images,labels, n_init, n_jobs=-1,
		save_dir='./results/screen_clustering/',
		learning_rate=0.01,
		momentum=0.9,
		pretrain_epochs=600,
		batch_size=200,
		maxiter=8000,
		update_interval=200,
		tol=0.001):
	x = images.reshape((images.shape[0], -1))
	y = labels
	n_clusters = len(np.unique(y))

	#pretrain autoencoder
	dims = [x.shape[-1], 500, 500, 2000, 6]
	init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform')
	pretrain_optimizer = SGD(lr=learning_rate, momentum=momentum)
	autoencoder1, encoder1 = autoencoder(dims, init=init)
	autoencoder1.compile(optimizer=pretrain_optimizer, loss='mse')
	try:
		autoencoder1.load_weights(save_dir + '/ae_weights.h5')
	except:
		autoencoder1.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
		autoencoder1.save_weights(save_dir + '/ae_weights.h5')
	
	clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder1.output)
	model = Model(inputs=encoder1.input, outputs=clustering_layer)
	model.compile(optimizer=SGD(learning_rate, momentum), loss='kld')

	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
	y_pred = kmeans.fit_predict(encoder1.predict(x))
	y_pred_last = np.copy(y_pred)
	model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

	def target_distribution(q):
	    weight = q ** 2 / q.sum(0)
	    return (weight.T / weight.sum(1)).T

	loss = 0
	index = 0
	index_array = np.arange(x.shape[0])

	# start training

	for ite in range(int(maxiter)):
	    if ite % update_interval == 0:
	        q = model.predict(x, verbose=0)
	        p = target_distribution(q)  # update the auxiliary target distribution p

	        # evaluate the clustering performance
	        y_pred = q.argmax(1)
	        if y is not None:
	            nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	            loss = np.round(loss, 5)
	            print('Iter %d: nmi = %.5f, ari = %.5f' % (ite, nmi, ari), ' ; loss=', loss)

	        # check stop criterion - model convergence
	        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
	        y_pred_last = np.copy(y_pred)
	        if ite > 0 and delta_label < tol:
	            print('delta_label ', delta_label, '< tol ', tol)
	            print('Reached tolerance threshold. Stopping training.')
	            break
	    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
	    loss = model.train_on_batch(x=x[idx], y=p[idx])
	    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

	model.save_weights(save_dir + '/DEC_model_final.h5')

	# Eval.
	q = model.predict(x, verbose=0)
	p = target_distribution(q)  # update the auxiliary target distribution p
	y_pred = q.argmax(1)
	if y is not None:
	    nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	    loss = np.round(loss, 5)
	    print('nmi = %.5f, ari = %.5f' % (nmi, ari), ' ; loss=', loss)
	return nmi, ari

def Conv_AutoEncoder_Kmeans(images,labels, n_init, n_jobs=-1,
		save_dir='./results/screen_clustering/',
		learning_rate=0.01,
		momentum=0.9,
		pretrain_epochs=600,
		batch_size=200,
		maxiter=8000,
		update_interval=200,
		tol=0.001,
		input_shape=(28,28,1),
		filters=[32, 64, 128, 10]):
	x = images
	y = labels
	n_clusters = len(np.unique(y))

	def autoencoderConv2D_1(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
	    input_img = Input(shape=input_shape)
	    if input_shape[0] % 8 == 0:
	        pad3 = 'same'
	    else:
	        pad3 = 'valid'
	    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

	    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

	    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)

	    x = Flatten()(x)
	    encoded = Dense(units=filters[3], name='embedding')(x)
	    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

	    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
	    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

	    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

	    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
	    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

	autoencoder2, encoder2 = autoencoderConv2D_1(input_shape=input_shape, filters=filters)
	autoencoder2.compile(optimizer='adadelta', loss='mse')
	autoencoder2.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
	autoencoder2.save_weights(save_dir+'/conv_ae_weights.h5')
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
	y_pred = kmeans.fit_predict(encoder2.predict(x))
	nmi = metrics.mutual_info_score(y,y_pred)
	ari = metrics.adjusted_rand_score(y,y_pred)
	print('nmi = %.5f, ari = %.5f' % (nmi, ari))
	return nmi, ari



def Conv_DEC_Kmeans(images,labels, n_init, n_jobs=-1,
		save_dir='./results/screen_clustering/',
		learning_rate=0.01,
		momentum=0.9,
		pretrain_epochs=600,
		batch_size=200,
		maxiter=8000,
		update_interval=200,
		tol=0.001,
		input_shape=(28,28,1),
		filters=[32, 64, 128, 10]):
	x = images
	y = labels
	n_clusters = len(np.unique(y))

	def autoencoderConv2D_1(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
	    input_img = Input(shape=input_shape)
	    if input_shape[0] % 8 == 0:
	        pad3 = 'same'
	    else:
	        pad3 = 'valid'
	    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

	    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

	    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)

	    x = Flatten()(x)
	    encoded = Dense(units=filters[3], name='embedding')(x)
	    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

	    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
	    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

	    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

	    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
	    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

	autoencoder2, encoder2 = autoencoderConv2D_1(input_shape=input_shape, filters=filters)
	autoencoder2.compile(optimizer='adadelta', loss='mse')
	try: 
		autoencoder2.load_weights(save_dir+'/conv_ae_weights.h5')
	except:
		autoencoder2.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
		autoencoder2.save_weights(save_dir+'/conv_ae_weights.h5')
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
	y_pred = kmeans.fit_predict(encoder2.predict(x))
	y_pred_last = np.copy(y_pred)
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder2.output)
    model = Model(inputs=encoder2.input, outputs=[clustering_layer, autoencoder2.output])
	model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
	loss = 0
	index = 0
	index_array = np.arange(x.shape[0])
	# start training
	for ite in range(int(maxiter)):
	    if ite % update_interval == 0:
	        q = model.predict(x, verbose=0)
	        p = target_distribution(q)  # update the auxiliary target distribution p
	        y_pred = q.argmax(1)
	        if y is not None:
	            nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	            loss = np.round(loss, 5)
	            print('Iter %d:nmi = %.5f, ari = %.5f' % (ite,nmi, ari), ' ; loss=', loss)

	        # check stop criterion
	        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
	        y_pred_last = np.copy(y_pred)
	        if ite > 0 and delta_label < tol:
	            print('delta_label ', delta_label, '< tol ', tol)
	            print('Reached tolerance threshold. Stopping training.')
	            break
	    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
	    loss = model.train_on_batch(x=x[idx], y=p[idx])
	    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

	model.save_weights(save_dir + '/conv_DEC_model_final.h5')
	# Eval.
	q = model.predict(x, verbose=0)
	p = target_distribution(q)  # update the auxiliary target distribution p
	y_pred = q.argmax(1)
	if y is not None:
	    nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	    loss = np.round(loss, 5)
	    print('nmi = %.5f, ari = %.5f' % (nmi, ari), ' ; loss=', loss)
	return nmi, ari

def Conv_b_DEC_Kmeans(images,labels, n_init, n_jobs=-1,
		save_dir='./results/screen_clustering/',
		learning_rate=0.01,
		momentum=0.9,
		pretrain_epochs=600,
		batch_size=200,
		maxiter=8000,
		update_interval=200,
		tol=0.001,
		input_shape=(28,28,1),
		filters=[32, 64, 128, 10]):
	x = images
	y = labels
	n_clusters = len(np.unique(y))
	x = images
	y = labels
	n_clusters = len(np.unique(y))

	def autoencoderConv2D_1(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
	    input_img = Input(shape=input_shape)
	    if input_shape[0] % 8 == 0:
	        pad3 = 'same'
	    else:
	        pad3 = 'valid'
	    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

	    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

	    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)

	    x = Flatten()(x)
	    encoded = Dense(units=filters[3], name='embedding')(x)
	    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

	    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
	    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

	    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

	    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
	    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

	autoencoder2, encoder2 = autoencoderConv2D_1(input_shape=input_shape, filters=filters)
	try:
		autoencoder2.load_weights(save_dir+'/conv_ae_weights.h5')
	except:
		autoencoder2.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
		autoencoder2.save_weights(save_dir+'/conv_ae_weights.h5')
	clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
	model = Model(inputs=encoder2.input, outputs=[clustering_layer, autoencoder2.output])
	kmeans = KMeans(n_clusters=n_clusters, n_init=20)
	y_pred = kmeans.fit_predict(encoder.predict(x))
	model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
	y_pred_last = np.copy(y_pred)

	loss = 0
	index = 0
	index_array = np.arange(x.shape[0])
	model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')
	# start training
	for ite in range(int(maxiter)):
	    if ite % update_interval == 0:
	        q, _  = model.predict(x, verbose=0)
	        p = target_distribution(q)  # update the auxiliary target distribution p

	        # evaluate the clustering performance
	        y_pred = q.argmax(1)
	        if y is not None:
	            nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	            loss = np.round(loss, 5)
	            print('Iter %d:nmi = %.5f, ari = %.5f' % (ite,nmi, ari), ' ; loss=', loss)

	        # check stop criterion
	        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
	        y_pred_last = np.copy(y_pred)
	        if ite > 0 and delta_label < tol:
	            print('delta_label ', delta_label, '< tol ', tol)
	            print('Reached tolerance threshold. Stopping training.')
	            break
	    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
	    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
	    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

	model.save_weights(save_dir + '/conv_b_DEC_model_final.h5')
	q, _ = model.predict(x, verbose=0)
	p = target_distribution(q)  # update the auxiliary target distribution p
	y_pred = q.argmax(1)
	if y is not None:
	    nmi = np.round(metrics.mutual_info_score(y, y_pred), 5)
	    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
	    loss = np.round(loss, 5)
	    print('nmi = %.5f, ari = %.5f' % (nmi, ari), ' ; loss=', loss)
	return nmi, ari



