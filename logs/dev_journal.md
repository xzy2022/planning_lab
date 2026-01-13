[17:18]  [ISSUE] [test_rrt_planning.py] 发现 set_offsets 报错 IndexError，原因为 State 对象列表未正确提取坐标 -> 修复方案：在转换为 numpy 数组前提取 x, y
[17:18]  [ATTEMPT] [尝试 1] 修改 tests 目录下 A* 和 RRT 动画脚本，在 np.array 转换时显式提取 [s.x, s.y] ->  成功: IndexError 消除，动画脚本可正常运行
[17:19]  [DISCOVERY] [test_rrt_planning.py] 发现函数名与打印信息中存在大量 A* 的遗留文案 (Copy-paste error)，已全部更正为 RRT
[17:19]  [ATTEMPT] [尝试 2] 将 test_rrt_planning.py 的碰撞检测方法从 CIRCLE_ONLY 改为 RASTER，以匹配地图生成时的精度 -> 验证是否能成功规划
[17:20]  [ATTEMPT] [尝试 2] 将 test_rrt_planning.py 碰撞检测改为 RASTER 并增加迭代次数 ->  成功: 规划成功，IndexError 已修复
