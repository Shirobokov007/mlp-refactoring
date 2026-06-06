"""
Многослойный перцептрон на чистом Python (без NumPy)
Рефакторинг оригинального кода из блокнота
"""

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    """Сигмоидная функция активации"""
    # Защита от переполнения
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


def sigmoid_derivative(x: float) -> float:
    """Производная сигмоидной функции"""
    s = sigmoid(x)
    return s * (1 - s)


class MLPPurePython:
    """
    Многослойный перцептрон на чистом Python (без внешних зависимостей).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.5):
        """
        Инициализация MLP.
        
        Аргументы:
            input_size: Количество входных нейронов
            hidden_size: Количество скрытых нейронов
            output_size: Количество выходных нейронов
            learning_rate: Скорость обучения
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов и смещений
        self._initialize_weights()
        
        # Временные массивы для кэширования
        self._init_caches()
    
    def _initialize_weights(self) -> None:
        """Инициализация весов случайными числами в диапазоне [-1, 1]"""
        # Веса скрытого слоя
        self.W1 = [
            [random.uniform(-1, 1) for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        self.b1 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        
        # Веса выходного слоя
        self.W2 = [
            [random.uniform(-1, 1) for _ in range(self.hidden_size)]
            for _ in range(self.output_size)
        ]
        self.b2 = [random.uniform(-1, 1) for _ in range(self.output_size)]
    
    def _init_caches(self) -> None:
        """Инициализация массивов для кэширования промежуточных значений"""
        self.hidden_net = [0.0] * self.hidden_size
        self.hidden_outputs = [0.0] * self.hidden_size
        self.final_net = [0.0] * self.output_size
        self.final_outputs = [0.0] * self.output_size
        self.hidden_errors = [0.0] * self.hidden_size
        self.final_errors = [0.0] * self.output_size
    
    def forward(self, x: List[float]) -> None:
        """
        Прямой проход.
        
        Аргументы:
            x: Входной вектор
        """
        # Скрытый слой
        for i in range(self.hidden_size):
            self.hidden_net[i] = sum(self.W1[i][j] * x[j] for j in range(self.input_size)) + self.b1[i]
            self.hidden_outputs[i] = sigmoid(self.hidden_net[i])
        
        # Выходной слой
        for k in range(self.output_size):
            self.final_net[k] = sum(self.W2[k][j] * self.hidden_outputs[j] for j in range(self.hidden_size)) + self.b2[k]
            self.final_outputs[k] = sigmoid(self.final_net[k])
    
    def backward(self, x: List[float], y_true: List[float]) -> None:
        """
        Обратный проход (backpropagation).
        
        Аргументы:
            x: Входной вектор
            y_true: Истинные значения
        """
        # Ошибка на выходном слое
        for k in range(self.output_size):
            error = y_true[k] - self.final_outputs[k]
            self.final_errors[k] = error * sigmoid_derivative(self.final_net[k])
        
        # Ошибка на скрытом слое
        for j in range(self.hidden_size):
            error_sum = sum(self.W2[k][j] * self.final_errors[k] for k in range(self.output_size))
            self.hidden_errors[j] = error_sum * sigmoid_derivative(self.hidden_net[j])
        
        # Обновление весов выходного слоя
        for k in range(self.output_size):
            for j in range(self.hidden_size):
                self.W2[k][j] += self.learning_rate * self.final_errors[k] * self.hidden_outputs[j]
            self.b2[k] += self.learning_rate * self.final_errors[k]
        
        # Обновление весов скрытого слоя
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                self.W1[j][i] += self.learning_rate * self.hidden_errors[j] * x[i]
            self.b1[j] += self.learning_rate * self.hidden_errors[j]
    
    def train(self, X: List[List[float]], Y: List[int], epochs: int = 10000, print_every: int = 1000) -> None:
        """
        Обучение MLP.
        
        Аргументы:
            X: Список входных векторов
            Y: Список целевых значений
            epochs: Количество эпох
            print_every: Частота печати потерь
        """
        for epoch in range(epochs):
            total_loss = 0.0
            
            for x, y_true_scalar in zip(X, Y):
                y_true = [float(y_true_scalar)]  # Преобразуем в список для единообразия
                
                self.forward(x)
                
                loss = sum((y_true[k] - self.final_outputs[k]) ** 2 for k in range(self.output_size))
                total_loss += loss
                
                self.backward(x, y_true)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def predict(self, x: List[float]) -> int:
        """
        Предсказание для одного примера.
        
        Аргументы:
            x: Входной вектор
        
        Возвращает:
            0 или 1
        """
        self.forward(x)
        return round(self.final_outputs[0])


def demo_mlp_pure():
    """Демонстрация работы MLP на чистом Python"""
    print("=" * 60)
    print("MLP на чистом Python — задача XOR")
    print("=" * 60)
    
    # Данные для XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 0]
    
    print("\nИсходные данные XOR:")
    for x, y in zip(X, Y):
        print(f"  {x} -> {y}")
    
    # Создание и обучение MLP
    print("\nОбучение MLP...")
    mlp = MLPPurePython(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
    mlp.train(X, Y, epochs=10000, print_every=1000)
    
    # Результаты
    print("\nРезультаты предсказания:")
    for x, y_true in zip(X, Y):
        pred = mlp.predict(x)
        print(f"  Вход: {x}, Истинное: {y_true}, Предсказанное: {pred}")


if __name__ == "__main__":
    demo_mlp_pure()