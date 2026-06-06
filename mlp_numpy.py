"""
Многослойный перцептрон (MLP) с обратным распространением ошибки
Рефакторинг кода из учебного блокнота
Решает задачу XOR (исключающее ИЛИ)
"""

import numpy as np
from typing import List, Tuple, Optional


class ActivationFunctions:
    """Класс с функциями активации и их производными"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Защита от переполнения
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Производная сигмоидной функции"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)


class MultiLayerPerceptron:
    """
    Многослойный перцептрон (MLP) с одним скрытым слоем.
    
    Алгоритм:
    1. Прямой проход (forward pass)
    2. Вычисление ошибки
    3. Обратный проход (backpropagation)
    4. Обновление весов и смещений
    
    Архитектура: Input -> Hidden -> Output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.5,
        random_seed: Optional[int] = None
    ):
        """
        Инициализация MLP.
        
        Аргументы:
            input_size: Размер входного слоя (количество признаков)
            hidden_size: Размер скрытого слоя (количество нейронов)
            output_size: Размер выходного слоя
            learning_rate: Скорость обучения
            random_seed: Seed для воспроизводимости результатов
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов и смещений (метод Xavier)
        self._initialize_weights()
        
        # Кэширование промежуточных значений для обратного прохода
        self._cache = {}
    
    def _initialize_weights(self) -> None:
        """Инициализация весов методом Xavier/Glorot"""
        # Веса скрытого слоя: hidden_size x input_size
        limit1 = np.sqrt(6 / (self.input_size + self.hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (self.hidden_size, self.input_size))
        self.b1 = np.zeros(self.hidden_size)
        
        # Веса выходного слоя: output_size x hidden_size
        limit2 = np.sqrt(6 / (self.hidden_size + self.output_size))
        self.W2 = np.random.uniform(-limit2, limit2, (self.output_size, self.hidden_size))
        self.b2 = np.zeros(self.output_size)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Прямой проход.
        
        Аргументы:
            X: Входные данные (n_samples x input_size)
        
        Возвращает:
            Выход сети (n_samples x output_size)
        """
        # Скрытый слой: Z1 = W1 * X + b1, A1 = sigmoid(Z1)
        Z1 = np.dot(self.W1, X.T).T + self.b1
        A1 = ActivationFunctions.sigmoid(Z1)
        
        # Выходной слой: Z2 = W2 * A1 + b2, A2 = sigmoid(Z2)
        Z2 = np.dot(self.W2, A1.T).T + self.b2
        A2 = ActivationFunctions.sigmoid(Z2)
        
        # Сохраняем для обратного прохода
        self._cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        
        return A2
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисление среднеквадратичной ошибки (MSE)"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Обратный проход (backpropagation).
        Обновление весов на основе градиента ошибки.
        """
        n_samples = X.shape[0]
        
        # Извлекаем кэшированные значения
        A1 = self._cache['A1']
        Z2 = self._cache['Z2']
        A2 = self._cache['A2']
        
        # Ошибка на выходном слое
        dZ2 = (A2 - y_true) * ActivationFunctions.sigmoid_derivative(Z2)
        dW2 = np.dot(dZ2.T, A1) / n_samples
        db2 = np.sum(dZ2, axis=0) / n_samples
        
        # Ошибка на скрытом слое
        dZ1 = np.dot(dZ2, self.W2) * ActivationFunctions.sigmoid_derivative(self._cache['Z1'])
        dW1 = np.dot(dZ1.T, X) / n_samples
        db1 = np.sum(dZ1, axis=0) / n_samples
        
        # Обновление весов и смещений
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10000,
        verbose: bool = True,
        print_every: int = 1000
    ) -> List[float]:
        """
        Обучение MLP.
        
        Аргументы:
            X: Матрица признаков (n_samples x input_size)
            y: Вектор целевых значений (n_samples,)
            epochs: Количество эпох обучения
            verbose: Печатать ли потери
            print_every: Частота печати потерь
        
        Возвращает:
            Список значений потерь по эпохам
        """
        # Преобразуем y в правильную форму (n_samples x output_size)
        y_reshaped = y.reshape(-1, 1)
        
        losses = []
        
        for epoch in range(epochs):
            # Прямой проход
            y_pred = self._forward(X)
            
            # Вычисление потерь
            loss = self._compute_loss(y_reshaped, y_pred)
            losses.append(loss)
            
            # Обратный проход
            self._backward(X, y_reshaped)
            
            # Печать прогресса
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Предсказание для набора данных.
        
        Аргументы:
            X: Входные данные (n_samples x input_size)
            threshold: Порог бинаризации (по умолчанию 0.5)
        
        Возвращает:
            Массив предсказаний (0 или 1)
        """
        y_pred = self._forward(X)
        return (y_pred >= threshold).flatten().astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности предсказаний"""
        return self._forward(X).flatten()


def create_xor_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Создание датасета для задачи XOR"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def demo_mlp():
    """Демонстрация работы MLP на задаче XOR"""
    print("=" * 60)
    print("Многослойный перцептрон (MLP) — задача XOR")
    print("=" * 60)
    
    # Создание датасета
    X, y = create_xor_dataset()
    
    print("\nДатасет XOR:")
    for x, target in zip(X, y):
        print(f"  Вход: {x}, Цель: {target}")
    
    # Создание и обучение MLP
    print("\nОбучение MLP...")
    mlp = MultiLayerPerceptron(
        input_size=2,
        hidden_size=2,
        output_size=1,
        learning_rate=0.5,
        random_seed=42
    )
    
    losses = mlp.fit(X, y, epochs=10000, verbose=True, print_every=1000)
    
    # Предсказания
    print("\nРезультаты предсказания:")
    predictions = mlp.predict(X)
    probabilities = mlp.predict_proba(X)
    
    for x, y_true, y_pred, prob in zip(X, y, predictions, probabilities):
        print(f"  Вход: {x}, Истинное: {y_true}, "
              f"Предсказанное: {y_pred}, Вероятность: {prob:.4f}")
    
    # Итоговая точность
    accuracy = np.mean(predictions == y)
    print(f"\nТочность: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    demo_mlp()