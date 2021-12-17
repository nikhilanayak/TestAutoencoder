import jax
from tqdm import tqdm
from jax import jit
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
from flax import optim
import loader
import optax
from flax.training import train_state


rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)


def conv_and_pool(layer, features, kernel_size):
	x = nn.Conv(features=features, kernel_size=kernel_size)(layer)
	x = nn.relu(x)
	x = nn.max_pool(x, window_shape=(1, 2, 2), strides=(1, 2, 2), padding="SAME")
	#print(x.shape)
	return x

def convtrans_layer(layer, features, kernel_size):
  x = nn.ConvTranspose(features, kernel_size=kernel_size, strides=(1, 2, 2), padding="SAME")(layer)
  x = nn.relu(x)
  return x

class Encoder(nn.Module):
	@nn.compact
	def __call__(self, input_layer):
		x = conv_and_pool(input_layer, features=2, kernel_size=(3, 3, 1))
		x = conv_and_pool(x, features=4, kernel_size=(3, 3, 1))
		x = conv_and_pool(x, features=8, kernel_size=(3, 3, 1))
		x = conv_and_pool(x, features=16, kernel_size=(3, 3, 1))
		x = conv_and_pool(x, features=32, kernel_size=(3, 3, 1))
		#print("encoder")
		#print(x.shape) # == 68, 33, 2, 32

		return x
  
class Decoder(nn.Module):

	@nn.compact
	def __call__(self, input_layer):
		x = convtrans_layer(input_layer, 32, kernel_size=(3, 3, 1))
		x = convtrans_layer(x, 16, kernel_size=(3, 3, 1))
		x = convtrans_layer(x, 8, kernel_size=(3, 3, 1))
		x = convtrans_layer(x, 4, kernel_size=(3, 3, 1))
		x = convtrans_layer(x, 2, kernel_size=(3, 3, 1))
		x = nn.Conv(1, (3, 3, 1), padding="SAME")(x)
		#print("decoder")
		#print(x.shape)
		return x

class Autoencoder(nn.Module):
	def setup(self):
		self.encoder = Encoder()
		self.decoder = Decoder()

	def __call__(self, x):
		#print(args)
		#print(x)
		#print(kwargs)
		out = self.encoder(x)
		reconstructed = self.decoder(out)
		#print(f"out: {out}, rec: {reconstructed.shape}")
		return reconstructed, x 
	
	#@nn.module_method
	def generate(self, x):
		params = self.get_param("decoder")
		return Decoder.call(params, x)


#@jit
def compute_metrics(generated, real):
	error = generated - real
	loss = jnp.mean(jnp.square(error))
	metrics = {
		"loss": loss
	}		
	return metrics


dataset = loader.Dataset(8)

#@jit
def train_step(state):
	def loss_fn(params):
		#data = jnp.zeros((2176, 1056, 2, 1)) # DATA ?!?!
		data = dataset[0][0]

		rec, out = Autoencoder().apply({"params": params}, data)
		#globals()["p"] = rec, out
		loss = compute_metrics(rec, out)["loss"]
		print(loss)
		return jnp.array(loss)

	#optimizer, _, _ = optimizer.optimize(loss_fn)
	grads = jax.grad(loss_fn)(state.params)
	return state.apply_gradients(grads=grads)
	#return optimizer
  
#@jax.jit
#def eval_step(params, batch):
#  logits = CNN().apply({'params': params}, batch['image'])
#  return compute_metrics(logits, batch['label'])
print(f"{jax.device_count()} devices")

#_, autoencoder = Autoencoder.create_by_shape(KEY, [((2176, 1056, 2, 1), jnp.float32)])

optimizer = optim.sgd.GradientDescent(learning_rate=0.01)
EPOCHS = 10

state = train_state.TrainState.create(
	apply_fn=Autoencoder().apply,
	params=Autoencoder().init(rng, dataset[0][0])["params"],
	tx=optax.sgd(0.01)
)

for epoch in range(EPOCHS):
	bar = tqdm(range(len(dataset)))
	for _ in bar:
		state = train_step(state)
		bar.set_description(f"Epoch {epoch}")

