import numpy as np
from dataclasses import dataclass, astuple


def compute_probabilities(logits):
    """Вычисляет вероятности из логитов"""
    maxvals = np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits - maxvals)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def calculate_loss(probs, labels):
    """Вычисляет кросс-энтропийную функцию потерь"""
    batch_size = probs.shape[0]
    indices = np.arange(batch_size)
    target_probs = probs[indices, labels.flatten()]
    return -np.sum(np.log(target_probs))


def loss_and_gradient(logits, labels):
    """Вычисляет функцию потерь и градиент"""
    probs = compute_probabilities(logits)
    loss = calculate_loss(probs, labels)

    grad = probs.copy()
    grad[np.arange(grad.shape[0]), labels.flatten()] -= 1

    return loss, grad


def weight_penalty(weights, lambda_reg):
    """L2 регуляризация"""
    penalty = lambda_reg * np.sum(weights[:-1] ** 2)
    grad = 2 * lambda_reg * weights
    grad[-1] = 0
    return penalty, grad


def verify_gradient(func, x, eps=1e-5, tolerance=1e-4):
    """Проверяет корректность аналитического градиента"""
    x_copy = x.copy()
    val, analytical = func(x)

    for idx in np.ndindex(x.shape):
        delta = np.zeros_like(x)
        delta[idx] = eps

        val_plus_delta, _ = func(x + delta)
        numeric = (val_plus_delta - val) / eps

        if not np.isclose(numeric, analytical[idx], rtol=tolerance):
            return False

    return True


def activation_forward(x):
    """ReLU активация"""
    return np.maximum(0, x)


def activation_backward(grad_output, input_val):
    """Обратное распространение через ReLU"""
    mask = input_val > 0
    return grad_output * mask


@dataclass
class NormalizationCache:
    input: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    normalized: np.ndarray
    scale: float
    shift: float
    eps: float

    def unpack(self):
        return astuple(self)


def normalize_batch(input_data, scale, shift, epsilon=1e-5):
    """Пакетная нормализация (forward pass)"""
    mu = np.mean(input_data, axis=0)
    sigma = np.var(input_data, axis=0)
    normalized = (input_data - mu) / np.sqrt(sigma + epsilon)
    output = scale * normalized + shift

    cache = NormalizationCache(input_data, mu, sigma, normalized, scale, shift, epsilon)
    return output, cache


def normalize_batch_backward(grad_output, cache):
    """Пакетная нормализация (backward pass)"""
    input_data, mu, sigma, normalized, scale, shift, eps = cache.unpack()
    batch_size = input_data.shape[0]

    grad_normalized = grad_output * scale
    grad_scale = np.sum(grad_output * normalized, axis=0)
    grad_shift = np.sum(grad_output, axis=0)

    std_inv = 1 / np.sqrt(sigma + eps)
    centered = input_data - mu

    grad_var = np.sum(grad_normalized * centered * (-0.5) * (sigma + eps) ** (-1.5), axis=0)
    grad_mu = np.sum(grad_normalized * (-std_inv), axis=0) + grad_var * np.mean(-2 * centered, axis=0)

    grad_input = (grad_normalized * std_inv) + (2 * grad_var * centered / batch_size) + (grad_mu / batch_size)

    return grad_input, grad_scale, grad_shift


class NeuralClassifier:
    def __init__(self, input_size, hidden_size, num_classes):
        self._weights1 = 0.001 * np.random.randn(input_size, hidden_size)
        self._weights2 = 0.001 * np.random.randn(hidden_size, num_classes)
        self._bias1 = np.zeros(hidden_size)
        self._bias2 = np.zeros(num_classes)

    def train(self, X, y, batch_size=100, alpha=1e-7, reg=1e-5, num_epochs=1):
        losses = []
        samples = X.shape[0]

        for epoch in range(num_epochs):
            perm = np.random.permutation(samples)
            batches = np.array_split(perm, np.arange(batch_size, samples, batch_size))

            for batch in batches:
                X_batch, y_batch = X[batch], y[batch]

                # Forward
                h1 = activation_forward(X_batch @ self._weights1 + self._bias1)
                scores = h1 @ self._weights2 + self._bias2

                loss, grad_scores = loss_and_gradient(scores, y_batch)
                reg_loss1, reg_grad1 = weight_penalty(self._weights1, reg)
                reg_loss2, reg_grad2 = weight_penalty(self._weights2, reg)
                loss += reg_loss1 + reg_loss2

                # Backward
                grad_h1 = grad_scores @ self._weights2.T
                grad_h1 = activation_backward(grad_h1, h1)

                grad_w2 = h1.T @ grad_scores + reg_grad2
                grad_w1 = X_batch.T @ grad_h1 + reg_grad1
                grad_b2 = grad_scores.sum(axis=0)
                grad_b1 = grad_h1.sum(axis=0)

                # Update
                self._weights1 -= alpha * grad_w1
                self._weights2 -= alpha * grad_w2
                self._bias1 -= alpha * grad_b1
                self._bias2 -= alpha * grad_b2

            losses.append(loss)

        return losses

    def predict_proba(self, X):
        h1 = activation_forward(X @ self._weights1 + self._bias1)
        scores = h1 @ self._weights2 + self._bias2
        return compute_probabilities(scores)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class BatchNormClassifier:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Инициализация весов и смещений
        self._W1 = 0.001 * np.random.randn(input_dim, hidden_dim)
        self._W2 = 0.001 * np.random.randn(hidden_dim, output_dim)
        self._b1 = np.zeros(hidden_dim)
        self._b2 = np.zeros(output_dim)

        # Параметры batch normalization
        self._scale = np.ones(hidden_dim)
        self._shift = np.zeros(hidden_dim)

    def train(self, X, y, batch_size=100, alpha=1e-7, reg=1e-5, num_epochs=1):
        losses = []
        samples = X.shape[0]

        for epoch in range(num_epochs):
            perm = np.random.permutation(samples)
            batches = np.array_split(perm, np.arange(batch_size, samples, batch_size))

            for batch in batches:
                X_batch, y_batch = X[batch], y[batch]

                # Forward pass
                h1 = X_batch @ self._W1 + self._b1
                h1_norm, cache = normalize_batch(h1, self._scale, self._shift)
                a1 = activation_forward(h1_norm)
                scores = a1 @ self._W2 + self._b2

                # Loss computation
                loss, grad_scores = loss_and_gradient(scores, y_batch)
                reg_loss1, reg_grad1 = weight_penalty(self._W1, reg)
                reg_loss2, reg_grad2 = weight_penalty(self._W2, reg)
                loss += reg_loss1 + reg_loss2

                # Backward pass
                grad_a1 = grad_scores @ self._W2.T
                grad_h1_norm = activation_backward(grad_a1, a1)
                grad_h1, grad_scale, grad_shift = normalize_batch_backward(grad_h1_norm, cache)

                grad_W2 = a1.T @ grad_scores + reg_grad2
                grad_W1 = X_batch.T @ grad_h1 + reg_grad1
                grad_b2 = grad_scores.sum(axis=0)
                grad_b1 = grad_h1.sum(axis=0)

                # Parameter updates
                self._W1 -= alpha * grad_W1
                self._W2 -= alpha * grad_W2
                self._b1 -= alpha * grad_b1
                self._b2 -= alpha * grad_b2
                self._scale -= alpha * grad_scale
                self._shift -= alpha * grad_shift

            losses.append(loss)
        return losses

    def predict_proba(self, X):
        h1 = X @ self._W1 + self._b1
        h1_norm, _ = normalize_batch(h1, self._scale, self._shift)
        a1 = activation_forward(h1_norm)
        scores = a1 @ self._W2 + self._b2
        return compute_probabilities(scores)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class MomentumBatchNormClassifier(BatchNormClassifier):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim)
        # Momentum buffers
        self._v_W1 = np.zeros_like(self._W1)
        self._v_W2 = np.zeros_like(self._W2)
        self._v_b1 = np.zeros_like(self._b1)
        self._v_b2 = np.zeros_like(self._b2)
        self._v_scale = np.zeros_like(self._scale)
        self._v_shift = np.zeros_like(self._shift)

    def train(self, X, y, batch_size=100, alpha=1e-7, reg=1e-5, num_epochs=1, momentum=0.9):
        losses = []
        samples = X.shape[0]

        for epoch in range(num_epochs):
            perm = np.random.permutation(samples)
            batches = np.array_split(perm, np.arange(batch_size, samples, batch_size))

            for batch in batches:
                X_batch, y_batch = X[batch], y[batch]

                # Forward pass
                h1 = X_batch @ self._W1 + self._b1
                h1_norm, cache = normalize_batch(h1, self._scale, self._shift)
                a1 = activation_forward(h1_norm)
                scores = a1 @ self._W2 + self._b2

                # Loss computation
                loss, grad_scores = loss_and_gradient(scores, y_batch)
                reg_loss1, reg_grad1 = weight_penalty(self._W1, reg)
                reg_loss2, reg_grad2 = weight_penalty(self._W2, reg)
                loss += reg_loss1 + reg_loss2

                # Backward pass
                grad_a1 = grad_scores @ self._W2.T
                grad_h1_norm = activation_backward(grad_a1, a1)
                grad_h1, grad_scale, grad_shift = normalize_batch_backward(grad_h1_norm, cache)

                grad_W2 = a1.T @ grad_scores + reg_grad2
                grad_W1 = X_batch.T @ grad_h1 + reg_grad1
                grad_b2 = grad_scores.sum(axis=0)
                grad_b1 = grad_h1.sum(axis=0)

                # Momentum updates
                self._v_W1 = momentum * self._v_W1 - alpha * grad_W1
                self._v_W2 = momentum * self._v_W2 - alpha * grad_W2
                self._v_b1 = momentum * self._v_b1 - alpha * grad_b1
                self._v_b2 = momentum * self._v_b2 - alpha * grad_b2
                self._v_scale = momentum * self._v_scale - alpha * grad_scale
                self._v_shift = momentum * self._v_shift - alpha * grad_shift

                self._W1 += self._v_W1
                self._W2 += self._v_W2
                self._b1 += self._v_b1
                self._b2 += self._v_b2
                self._scale += self._v_scale
                self._shift += self._v_shift

            losses.append(loss)
        return losses


class AdamBatchNormClassifier(BatchNormClassifier):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim)
        # Adam buffers
        self._m_W1 = np.zeros_like(self._W1)
        self._m_W2 = np.zeros_like(self._W2)
        self._m_b1 = np.zeros_like(self._b1)
        self._m_b2 = np.zeros_like(self._b2)
        self._m_scale = np.zeros_like(self._scale)
        self._m_shift = np.zeros_like(self._shift)

        self._v_W1 = np.zeros_like(self._W1)
        self._v_W2 = np.zeros_like(self._W2)
        self._v_b1 = np.zeros_like(self._b1)
        self._v_b2 = np.zeros_like(self._b2)
        self._v_scale = np.zeros_like(self._scale)
        self._v_shift = np.zeros_like(self._shift)

    def train(self, X, y, batch_size=100, alpha=1e-7, reg=1e-5, num_epochs=1, beta1=0.9, beta2=0.999, eps=1e-8):
        losses = []
        samples = X.shape[0]

        for epoch in range(num_epochs):
            perm = np.random.permutation(samples)
            batches = np.array_split(perm, np.arange(batch_size, samples, batch_size))

            for batch in batches:
                X_batch, y_batch = X[batch], y[batch]

                # Forward pass
                h1 = X_batch @ self._W1 + self._b1
                h1_norm, cache = normalize_batch(h1, self._scale, self._shift)
                a1 = activation_forward(h1_norm)
                scores = a1 @ self._W2 + self._b2

                # Loss computation
                loss, grad_scores = loss_and_gradient(scores, y_batch)
                reg_loss1, reg_grad1 = weight_penalty(self._W1, reg)
                reg_loss2, reg_grad2 = weight_penalty(self._W2, reg)
                loss += reg_loss1 + reg_loss2

                # Backward pass
                grad_a1 = grad_scores @ self._W2.T
                grad_h1_norm = activation_backward(grad_a1, a1)
                grad_h1, grad_scale, grad_shift = normalize_batch_backward(grad_h1_norm, cache)

                grad_W2 = a1.T @ grad_scores + reg_grad2
                grad_W1 = X_batch.T @ grad_h1 + reg_grad1
                grad_b2 = grad_scores.sum(axis=0)
                grad_b1 = grad_h1.sum(axis=0)

                # Adam updates
                def adam_update(param, m, v, grad):
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    param -= alpha * m / (np.sqrt(v) + eps)
                    return param, m, v

                self._W1, self._m_W1, self._v_W1 = adam_update(self._W1, self._m_W1, self._v_W1, grad_W1)
                self._W2, self._m_W2, self._v_W2 = adam_update(self._W2, self._m_W2, self._v_W2, grad_W2)
                self._b1, self._m_b1, self._v_b1 = adam_update(self._b1, self._m_b1, self._v_b1, grad_b1)
                self._b2, self._m_b2, self._v_b2 = adam_update(self._b2, self._m_b2, self._v_b2, grad_b2)
                self._scale, self._m_scale, self._v_scale = adam_update(self._scale, self._m_scale, self._v_scale,
                                                                        grad_scale)
                self._shift, self._m_shift, self._v_shift = adam_update(self._shift, self._m_shift, self._v_shift,
                                                                        grad_shift)

            losses.append(loss)
        return losses
