# src/collision/geometry.py
import numpy as np
import math

def check_circle_overlap(c1: tuple, r1: float, c2: tuple, r2: float) -> bool:
    """两圆是否相交"""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist_sq = dx*dx + dy*dy
    radius_sum = r1 + r2
    return dist_sq <= radius_sum * radius_sum

def check_sat_polygon_collision(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    使用分离轴定理 (SAT) 检测两个凸多边形是否相交
    :param poly1: (N, 2) 顶点数组
    :param poly2: (M, 2) 顶点数组
    :return: True if collision
    """
    for polygon in [poly1, poly2]:
        for i in range(len(polygon)):
            # 1. 获取分离轴 (边的法向量)
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            edge = p2 - p1
            # 法向量: (-y, x)
            axis = np.array([-edge[1], edge[0]])
            # 归一化 (可选，如果不归一化，投影长度会缩放，但不影响比较结果)
            # 为了数值稳定性，建议归一化
            norm = np.linalg.norm(axis)
            if norm < 1e-6: continue
            axis /= norm
            
            # 2. 投影
            min1, max1 = _project_polygon(axis, poly1)
            min2, max2 = _project_polygon(axis, poly2)
            
            # 3. 判断间隙
            if max1 < min2 or max2 < min1:
                return False # 找到分离轴，一定不相交
    return True

def _project_polygon(axis, poly):
    """辅助函数：将多边形投影到轴上"""
    dots = np.dot(poly, axis)
    return np.min(dots), np.max(dots)

def get_grid_aabb_polygon(x_idx: int, y_idx: int, resolution: float) -> np.ndarray:
    """
    将网格单元转换为矩形多边形顶点 (用于 SAT 检测)
    """
    x0 = x_idx * resolution
    y0 = y_idx * resolution
    x1 = x0 + resolution
    y1 = y0 + resolution
    # 顺时针或逆时针均可
    return np.array([
        [x0, y0], [x1, y0], [x1, y1], [x0, y1]
    ])