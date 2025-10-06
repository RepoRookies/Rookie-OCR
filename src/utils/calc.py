import re
import cv2
import numpy as np


class CalcUtil:
    @staticmethod
    def Normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalizes an image to the range [0, 1]
        Args:
            image (np.ndarray, 2D): The input image
        Returns:
            image (np.ndarray, 2D): The normalized image
        """
        return image / 255.0

    @staticmethod
    def Convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies a convolution to an image using a given kernel
        Args:
            image (np.ndarray, 2D): The input image
            kernel (np.ndarray, 2D): The convolution kernel
        Returns:
            image (np.ndarray, 2D): The convolved image
        """
        flipped_kernel = cv2.flip(kernel, -1)
        return cv2.filter2D(image, -1, flipped_kernel)

    @staticmethod
    def Correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies a correlation to an image using a given kernel
        Args:
            image (np.ndarray, 2D): The input image
            kernel (np.ndarray, 2D): The correlation kernel
        Returns:
            image (np.ndarray, 2D): The correlated image
        """
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def CosineSimilarity(first_string: str, second_string: str) -> float:
        """
        Calculates the cosine similarity between two strings
        Args:
            first (str): The first string
            second (str): The second string
        Returns:
            similarity (float): The cosine similarity between the two strings
        """
        first_vec = np.array([ord(c) for c in first_string])
        second_vec = np.array([ord(c) for c in second_string])

        max_len = max(len(first_vec), len(second_vec))
        first_vec = np.pad(first_vec, (0, max_len - len(first_vec)))
        second_vec = np.pad(second_vec, (0, max_len - len(second_vec)))

        return np.dot(first_vec, second_vec) / (
            np.linalg.norm(first_vec) * np.linalg.norm(second_vec)
        )

    @staticmethod
    def HammingDistance(first_string: str, second_string: str) -> int:
        """
        Calculates the Hamming distance between two strings, ignoring extra whitespaces.
        Args:
            first_string (str): The first string
            second_string (str): The second string
        Returns:
            distance (int): The Hamming distance between the two strings
        """

        first_string = re.sub(r"\s+", " ", first_string.strip())
        second_string = re.sub(r"\s+", " ", second_string.strip())

        max_len = max(len(first_string), len(second_string))
        first_string = first_string.ljust(max_len, " ")
        second_string = second_string.ljust(max_len, " ")

        return sum(1 for a, b in zip(first_string, second_string) if a != b)

    @staticmethod
    def LevenshteinDistance(first_string: str, second_string: str) -> int:
        """
        Calculates the Levenshtein distance between two strings, ignoring extra whitespaces.
        Args:
            first_string (str): The first string
            second_string (str): The second string
        Returns:
            distance (int): The Levenshtein distance between the two strings
        """
        first_string = re.sub(r"\s+", " ", first_string.strip())
        second_string = re.sub(r"\s+", " ", second_string.strip())

        m, n = len(first_string), len(second_string)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if first_string[i - 1] == second_string[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    )

        return dp[m][n]
