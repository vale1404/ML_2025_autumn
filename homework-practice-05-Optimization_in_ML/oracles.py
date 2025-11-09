import numpy as np
import scipy
from scipy.special import expit

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
       func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
    matvec_Ax : function
        Computes matrix-vector product Ax, where x is a vector of size n.
    matvec_ATx : function of x
        Computes matrix-vector product A^Tx, where x is a vector of size m.
    matmat_ATsA : function
        Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        """
        Вычисление значения функции логистической регрессии с L2-регуляризацией.
        
        f(x) = 1/m * sum(log(1 + exp(-b_i * <a_i, x>))) + regcoef/2 * ||x||^2
        """
        # Количество объектов
        m = self.b.shape[0]
        
        # Вычисляем A @ x
        Ax = self.matvec_Ax(x)
        
        # Вычисляем отступы: -b * (A @ x)
        margins = -self.b * Ax
        
        # Логистические потери с использованием logaddexp для численной стабильности
        # log(1 + exp(z)) = logaddexp(0, z)
        logistic_loss = np.mean(np.logaddexp(0, margins))
        
        # L2-регуляризация
        l2_reg = 0.5 * self.regcoef * np.dot(x, x)
        
        return logistic_loss + l2_reg

    def grad(self, x):
        """
        Вычисление градиента функции.
        
        grad = -1/m * A^T @ (b * sigmoid(-b * A @ x)) + regcoef * x
        где sigmoid(z) = 1 / (1 + exp(-z))
        """
        # Количество объектов
        m = self.b.shape[0]
        
        # Вычисляем A @ x
        Ax = self.matvec_Ax(x)
        
        # Вычисляем отступы: -b * (A @ x)
        margins = -self.b * Ax
        
        # Вычисляем sigmoid(-b * A @ x) используя expit для стабильности
        sigmoid_margins = expit(margins)
        
        # Взвешенная сигмоида: b * sigmoid(-b * A @ x)
        weighted_sigmoid = self.b * sigmoid_margins
        
        # Градиент логистических потерь
        grad_logistic = -(1.0 / m) * self.matvec_ATx(weighted_sigmoid)
        
        # Градиент регуляризации
        grad_reg = self.regcoef * x
        
        return grad_logistic + grad_reg

    def hess(self, x):
        """
        Вычисление гессиана функции.
        
        hess = 1/m * A^T @ diag(s) @ A + regcoef * I
        где s_i = sigmoid(-b_i * <a_i, x>) * (1 - sigmoid(-b_i * <a_i, x>))
        """
        # Количество объектов
        m = self.b.shape[0]
        
        # Вычисляем A @ x
        Ax = self.matvec_Ax(x)
        
        # Вычисляем отступы: -b * (A @ x)
        margins = -self.b * Ax
        
        # Вычисляем sigmoid(-b * A @ x)
        sigmoid_margins = expit(margins)
        
        # Диагональные элементы: s = sigmoid * (1 - sigmoid)
        s = sigmoid_margins * (1 - sigmoid_margins)
        
        # Гессиан логистических потерь
        hess_logistic = (1.0 / m) * self.matmat_ATsA(s)
        
        # Добавляем регуляризацию: regcoef * I
        n = x.shape[0]
        if scipy.sparse.issparse(hess_logistic):
            hess_reg = scipy.sparse.eye(n, format='csr') * self.regcoef
            # ← IMPORTANTE: Convierte a dense para tests
            return (hess_logistic + hess_reg).toarray()
        else:
            hess_reg = np.eye(n) * self.regcoef
            return hess_logistic + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle with caching for optimal performance.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        # Cache for x
        self._cache_x = None
        self._cache_Ax = None
        self._cache_margins = None
        self._cache_sigmoid = None
        # Cache for directional (d)
        self._cache_d = None
        self._cache_Ad = None
        # Cache for evaluated points in directional
        self._directional_cache = {}  # {tuple(x_eval): Ax_eval}

    def _update_cache_x(self, x):
        """Update cache for x."""
        # First check if this x was computed in directional
        x_tuple = tuple(x)
        if x_tuple in self._directional_cache:
            self._cache_x = np.copy(x)
            self._cache_Ax = self._directional_cache[x_tuple]
            self._cache_margins = -self.b * self._cache_Ax
            self._cache_sigmoid = expit(self._cache_margins)
            return
        
        # Otherwise compute normally
        if self._cache_x is None or not np.array_equal(x, self._cache_x):
            self._cache_x = np.copy(x)
            self._cache_Ax = self.matvec_Ax(x)
            self._cache_margins = -self.b * self._cache_Ax
            self._cache_sigmoid = expit(self._cache_margins)

    def _update_cache_d(self, d):
        """Update cache for d."""
        if self._cache_d is None or not np.array_equal(d, self._cache_d):
            self._cache_d = np.copy(d)
            self._cache_Ad = self.matvec_Ax(d)

    def func(self, x):
        """Function with caching."""
        self._update_cache_x(x)
        
        m = self.b.shape[0]
        logistic_loss = np.mean(np.logaddexp(0, self._cache_margins))
        l2_reg = 0.5 * self.regcoef * np.dot(x, x)
        
        return logistic_loss + l2_reg

    def grad(self, x):
        """Gradient with caching."""
        self._update_cache_x(x)
        
        m = self.b.shape[0]
        weighted_sigmoid = self.b * self._cache_sigmoid
        grad_logistic = -(1.0 / m) * self.matvec_ATx(weighted_sigmoid)
        grad_reg = self.regcoef * x
        
        return grad_logistic + grad_reg

    def hess(self, x):
        """Hessian with caching."""
        self._update_cache_x(x)
        
        m = self.b.shape[0]
        s = self._cache_sigmoid * (1 - self._cache_sigmoid)
        hess_logistic = (1.0 / m) * self.matmat_ATsA(s)
        
        n = x.shape[0]
        if scipy.sparse.issparse(hess_logistic):
            hess_reg = scipy.sparse.eye(n, format='csr') * self.regcoef
            return (hess_logistic + hess_reg).toarray()
        else:
            hess_reg = np.eye(n) * self.regcoef
            return hess_logistic + hess_reg

    def func_directional(self, x, d, alpha):
        """Optimized func_directional with caching."""
        self._update_cache_x(x)
        self._update_cache_d(d)
        
        Ax = self._cache_Ax
        Ad = self._cache_Ad
        
        m = self.b.shape[0]
        
        # A(x + alpha*d) = Ax + alpha*Ad
        A_x_alpha_d = Ax + alpha * Ad
        
        # Store in directional cache for potential future use
        x_eval = x + alpha * d
        self._directional_cache[tuple(x_eval)] = A_x_alpha_d
        
        margins = -self.b * A_x_alpha_d
        logistic_loss = np.mean(np.logaddexp(0, margins))
        
        l2_reg = 0.5 * self.regcoef * np.dot(x_eval, x_eval)
        
        return np.squeeze(logistic_loss + l2_reg)

    def grad_directional(self, x, d, alpha):
        """Optimized grad_directional with caching."""
        self._update_cache_x(x)
        self._update_cache_d(d)
        
        Ax = self._cache_Ax
        Ad = self._cache_Ad
        
        m = self.b.shape[0]
        
        # A(x + alpha*d) = Ax + alpha*Ad
        A_x_alpha_d = Ax + alpha * Ad
        
        # Store in directional cache
        x_eval = x + alpha * d
        self._directional_cache[tuple(x_eval)] = A_x_alpha_d
        
        margins = -self.b * A_x_alpha_d
        sigmoid_margins = expit(margins)
        weighted_sigmoid = self.b * sigmoid_margins
        
        grad_logistic_d = -(1.0 / m) * np.dot(Ad, weighted_sigmoid)
        grad_reg_d = self.regcoef * np.dot(x_eval, d)
        
        return np.squeeze(grad_logistic_d + grad_reg_d)




def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Вспомогательная функция для создания оракула логистической регрессии.
    
    Параметры
    ----------
    A : np.array или scipy.sparse matrix
        Матрица признаков (m x n).
    b : np.array
        Вектор меток классов (±1).
    regcoef : float
        Коэффициент L2-регуляризации.
    oracle_type : str
        Тип оракула: 'usual' или 'optimized'.
    
    Возвращает
    ----------
    oracle : LogRegL2Oracle или LogRegL2OptimizedOracle
    """
    # Определяем функции умножения в зависимости от типа матрицы A
    if scipy.sparse.issparse(A):
        # Для разреженных матриц
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        
        def matmat_ATsA(s):
            # A^T @ diag(s) @ A для разреженной матрицы
            return A.T.dot(scipy.sparse.diags(s).dot(A))
    else:
        # Для плотных матриц
        matvec_Ax = lambda x: A @ x
        matvec_ATx = lambda x: A.T @ x
        
        def matmat_ATsA(s):
            # A^T @ diag(s) @ A для плотной матрицы
            return A.T @ np.diag(s) @ A
    
    # Выбираем тип оракула
    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Вычисление приближённого градиента методом конечных разностей.
    
    Использует формулу:
    result_i := (f(x + eps * e_i) - f(x)) / eps,
    где e_i - координатные векторы.
    
    Параметры
    ----------
    func : callable
        Функция для вычисления градиента.
    x : np.array
        Точка, в которой вычисляется градиент.
    eps : float
        Шаг для конечных разностей.
    
    Возвращает
    ----------
    grad : np.array
        Приближённый градиент.
    """
    n = x.size
    grad = np.zeros(n)
    
    # Значение функции в точке x
    f_x = func(x)
    
    # Вычисляем градиент для каждой координаты
    for i in range(n):
        # Создаём единичный вектор e_i
        e_i = np.zeros(n)
        e_i[i] = 1.0
        
        # Односторонняя разность: (f(x + eps*e_i) - f(x)) / eps
        grad[i] = (func(x + eps * e_i) - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Вычисление приближённого гессиана методом конечных разностей.
    
    Использует формулу:
    result_{ij} := (f(x + eps*e_i + eps*e_j) - f(x + eps*e_i) 
                    - f(x + eps*e_j) + f(x)) / eps^2,
    где e_i - координатные векторы.
    
    Параметры
    ----------
    func : callable
        Функция для вычисления гессиана.
    x : np.array
        Точка, в которой вычисляется гессиан.
    eps : float
        Шаг для конечных разностей.
    
    Возвращает
    ----------
    hess : np.array
        Приближённый гессиан (симметричная матрица n x n).
    """
    n = x.size
    hess = np.zeros((n, n))
    
    # Значение функции в точке x
    f_x = func(x)
    
    # Предварительно вычисляем f(x + eps*e_i) для всех i
    f_x_plus = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_x_plus[i] = func(x + eps * e_i)
    
    # Вычисляем гессиан
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        
        for j in range(i, n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            
            # Вычисляем f(x + eps*e_i + eps*e_j)
            f_x_plus_ij = func(x + eps * e_i + eps * e_j)
            
            # Вторая смешанная производная
            hess[i, j] = (f_x_plus_ij - f_x_plus[i] - f_x_plus[j] + f_x) / (eps ** 2)
            
            # Используем симметричность гессиана
            hess[j, i] = hess[i, j]
    
    return hess
