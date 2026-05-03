import numpy as np
from numpy.lib.stride_tricks import as_strided


class Conv3x3:
    # Convolution layer using 3x3 filters.
    def __init__(self, num_filters, input_channels):
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters, 3, 3, input_channels).astype(np.float32) / np.sqrt(27)

    def iterate_regions(self, image):
        # Generates all possible 3x3 image regions
        h, w, _ = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input  # saved for backpropagation
        h, w, c = input.shape
        out_h, out_w = h - 2, w - 2
        # Use stride tricks to create a view of all 3x3xC patches
        shape = (out_h, out_w, 3, 3, c)
        strides = (
            input.strides[0],
            input.strides[1],
            input.strides[0],
            input.strides[1],
            input.strides[2],
        )
        patches = as_strided(input, shape=shape, strides=strides)
        # Compute convolution via einsum (sum over 3,3,c for each filter)
        output = np.einsum('ijabc,nabc->ijn', patches, self.filters)
        # output shape: (out_h, out_w, num_filters)
        return output.astype(np.float32)

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Gradient w.r.t filter: dL/df = dL/dout * dout/df (= im_region)
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPool2:
    # Max Pooling layer using a 2x2 pool size.
    def iterate_regions(self, image):
        h, w, f = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                # jump over already pooled pixels
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input  # save for backpropagation
        h, w, num_filters = input.shape
        new_h = h // 2
        new_w = w // 2
        # Reshape and pool
        out = input[:new_h*2, :new_w*2, :].reshape(new_h, 2, new_w, 2, num_filters)
        output = out.max(axis=(1, 3))
        return output.astype(np.float32)

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            # recalculate which value was the max
            amax = np.max(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # Only the pixel that was the max gets the gradient ( blame )
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class CNN:
    def __init__(self, num_filters=16, lr=0.001, verbose=False, input_shape=(32, 32, 3), epochs=100):
        """
        - num_filters: number of convolution filters
        - lr: learning rate
        - verbose: print loss during training
        - input_shape: shape of input images (e.g., (32,32,3) or (128,128,3))
        - epochs: number of training epochs
        """
        self.input_shape = input_shape
        self.conv = Conv3x3(num_filters, input_shape[2])
        self.pool = MaxPool2()
        # Dynamically compute dense_input_size based on input_shape
        h, w, c = input_shape
        h = h - 2  # after 3x3 conv
        w = w - 2
        h = h // 2  # after 2x2 pool
        w = w // 2
        self.dense_input_size = h * w * num_filters
        self.lr = lr
        self.verbose = verbose
        self.epochs = epochs

        # Xavier init
        self.w_dense = np.random.randn(self.dense_input_size, 1) / np.sqrt( self.dense_input_size)
        self.b_dense = np.zeros((1,), dtype=float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float32)

    def forward(self, out):
        """
        Forward pass for a single image (flat or 3D array).
        Always applies conv, ReLU, pool, flatten, then dense layer.
        """
        if out.ndim == 1:
            out = out.reshape(self.input_shape)

        # convolution -> ReLU -> pooling
        conv_out = self.conv.forward(out)
        self.last_conv_out = conv_out  # Save for ReLU backprop
        self.last_act_out = self.relu(conv_out)
        pool_out = self.pool.forward(self.last_act_out)

        self.last_pool_shape = pool_out.shape
        flat = pool_out.flatten().reshape(1, -1)

        # dense layer
        z = np.dot(flat, self.w_dense) + self.b_dense
        prediction = self.sigmoid(z)
        return prediction, flat

    def fit(self, X, y):
        """
        X: (n_samples, n_features) flattened images
        y: (n_samples, 1) labels
        """
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X, y = X[indices], y[indices]
            total_loss = 0

            for i in range(n_samples):
                # forward pass
                prediction, flat_input = self.forward(X[i])

                # gradient for output (BCE)
                d_output = y[i] - prediction
                total_loss += (d_output ** 2)

                # dense layer gradients
                d_w_dense = np.dot(flat_input.T, d_output)
                d_b_dense = np.sum(d_output, axis=0)

                # update dense weights immediately
                self.w_dense += self.lr * d_w_dense
                self.b_dense += self.lr * d_b_dense

                # backprop to pool layer
                d_L_d_pool = np.dot(d_output, self.w_dense.T).reshape(self.last_pool_shape)
                d_L_d_act = self.pool.backward(d_L_d_pool)

                # backprop through ReLU
                d_L_d_relu = d_L_d_act * self.relu_derivative(self.last_conv_out)

                # manually get filter gradients for img
                d_filters = np.zeros_like(self.conv.filters)
                for im_region, row, col in self.conv.iterate_regions(self.conv.last_input):
                    for f in range(self.conv.num_filters):
                        d_filters[f] += d_L_d_relu[row, col, f] * im_region

                # update conv weights immediately
                self.conv.filters += self.lr * d_filters

            if self.verbose:
                print(f"Epoch {epoch}, Average Loss: {np.mean(total_loss) / n_samples} Average Accuracy: {np.mean((self.predict(X) == y).astype(float))}")

    def predict(self, X, threshold=0.5):
        preds = []
        for xi in X:
            prob, _ = self.forward(xi)
            preds.append(1 if prob >= threshold else 0)
        return np.array(preds).reshape(-1, 1)
