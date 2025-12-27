# leetcode 215. 数组中第k个最大元素
# 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

# 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

# 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

# 要找到数组中第 k 个最大的元素，可以使用快速选择算法（Quickselect），这是快速排序的变种，平均时间复杂度为 O(n)。快速选择的基本思想是选择一个枢轴元素（pivot），将数组分为两部分：一部分小于枢轴，另一部分大于枢轴。根据枢轴的位置与 k 的关系，决定继续在哪一部分进行查找。

import random
from typing import List
def findKthLargest(nums: List[int], k: int) -> int:
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        store_index = left
        for i in range(left, right):
            if nums[i]>pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index] 
                store_index += 1
        nums[right], nums[store_index]  = nums[store_index], nums[right]  
        return store_index
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)
    return select(0, len(nums)-1, k-1)

# test_input = [3,2,1,5,6,4]
# res = findKthLargest(test_input, 2)
# print(res)  # 输出 5

# leetcode 207. 课程表
# 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

# 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程 bi 。

# 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
# 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

# 为了判断课程安排图中是否存在环（即是否可以完成所有课程），我们可以使用深度优先搜索（DFS）来检测环。以下是 DFS 方法的详细解释：

# 核心思想
# 将课程及其先修关系看作有向图，其中课程是节点，先修关系是有向边。

# 如果在 DFS 遍历过程中遇到一个正在被访问的节点（即当前 DFS 路径中已经访问过的节点），则说明存在环。

# 如果 DFS 完成所有节点的遍历都没有遇到这种情况，则说明无环，可以完成所有课程。

# 状态标记
# 为了区分节点的访问状态，我们使用一个 visited 数组，其中：

# 0：未被访问（初始状态）。

# 1：正在访问（当前 DFS 路径中）。

# 2：已访问完毕（从该节点出发的 DFS 已完成）。

def canFinish(numCourses, prerequisites):
    # 构建邻接表
    adj = [[] for _ in range(numCourses)]
    for i, j in prerequisites:
        adj[j].append(i)
    visited = [0] * numCourses
    def hasCycle(node):
        if visited[node] == 1: # 表示当前节点正在访问
            return True
        if visited[node] == 2: # 表示当前节点已经访问完毕
            return False
        visited[node] = 1
        for neighbor in adj[node]:
            if hasCycle(neighbor):
                return True
        visited[node] = 2
        return False
    for course in range(numCourses):
        if hasCycle(course):
            return False
    return True

# test_numCourses = 3
# test_prerequisites = [[1,0],[1,2]]
# res = canFinish(test_numCourses, test_prerequisites)
# print(res)  # 输出 False


# leetcode 206: 反转链表
# 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

#leetcode 200: 岛屿数量
# 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

# 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

# 此外，你可以假设该网格的四条边均被水包围。

def numIslands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    def dfs(i,j):
        if i<0 or j<0 or i>=rows or j>=cols or grid[i][j]=='0':
            return
        grid[i][j] = '0'
        dfs(i+1,j)
        dfs(i-1,j)
        dfs(i,j+1)
        dfs(i,j-1)

    island_count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count +=1
                dfs(i,j)
    return island_count

# leetcode 198 : 打家劫舍   
# 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
# 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

def rob(nums: List[int]) -> int:
    # dp[i] 表示 偷到第 i 个房屋时的最大金额
    if not nums:
        return 0
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    if n > 1:
        dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]

# leetcode 238 除自身以外数组的乘积 
# 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    answer = [1]*n
    # 计算左侧乘积
    for i in range(1, n):
        answer[i] = answer[i-1] * nums[i-1]
    # 计算右侧乘积并与左侧乘积相乘
    suffix_product = 1
    for i in range(n-1, -1, -1):
        answer[i] *= suffix_product
        suffix_product *= nums[i]
    return answer    


# leetcode 152 乘积最大子数组
# 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。示例 1:

# 输入: nums = [2,3,-2,4]
# 输出: 6
# 解释: 子数组 [2,3] 有最大乘积 6。

def maxProduct(nums: List[int])->int:
    if not nums:
        return 0
    n = len(nums)
    max_prod = min_prod = result = nums[0]
    for i in range(1, n):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(max_prod, min_prod)
    return result