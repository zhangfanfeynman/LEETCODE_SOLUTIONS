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

# leetcode 148. 排序链表
# 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
def sortList(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    # 找到中点
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None

    left = sortList(head)
    right = sortList(mid)
    # 合并两个有序链表
    return merge(left, right)
def merge(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 if l1 else l2
    return dummy.next

# leetcode 139: 单词拆分：给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。
# 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

# 你可以使用动态规划来解决这个问题。帮我生成一下动态规划的思路：
# 1. 定义一个布尔数组 dp，其中 dp[i] 表示字符串 s 的前 i 个字符是否可以被拆分成字典中的单词。
# 2. 初始化 dp[0] 为 True，因为空字符串可以被拆分。
# 3. 遍历字符串 s，对于每个位置 i，检查所有可能的前缀 s[j:i] 是否在字典中，并且 dp[j] 为 True。
# 4. 如果找到这样的 j，则 dp[i] 为 True。
# 5. 最后返回 dp[len(s)] 的值。
def workBreak(s:str, wordDict:List[str])->bool:
    word_set = set(wordDict)
    n = len(s)
    dp = [False]*(n+1)
    dp[0] = True
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]

# leetcode 647: 回文子串：
# 回文字符串 是正着读和倒过来读一样的字符串。

def countSubstrings(s:str)->int:
    n = len(s)
    count = 0
    for i in range(n):
        left, right = i,i
        while left>=0 and right<n and s[left]==s[right]:
            count +=1
            left -=1
            right +=1
        left, right = i, i+1
        while left>=0 and right<n and s[left]==s[right]:
            count +=1
            left -=1
            right +=1
    return count

# leetcode 128, 最长连续序列
# 给定一个未排序的整数数组 nums ，找出数字连续的最长序列的长度。
def longestConsecutive(nums: List[int])->int:
    if not nums:
        return 0
    nums.sort()
    max_length = current_length = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] +1:
            current_length +=1
            max_length = max(max_length, current_length)
        elif nums[i] == nums[i-1]:
            continue
        else:
            current_length = 1
    return max_length

# leetcode 322 零钱兑换
# 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

# 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
def coinChange(coins:List[int], amount:int)->int:
    dp = [float('inf')] * (amount+1)
    dp[0] = 0
    for i in range(1, amount+1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount] if dp[amount]!= float('inf') else -1

# leetcode 494 目标和
# 给你一个非负整数数组 nums 和一个整数 target 。

# 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

# 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
# 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

# 将添加 + 的数字分为一组（和为 S_plus），添加 - 的数字分为另一组（和为 S_minus）。

# 则有 S_plus - S_minus = target。

# 又因为 S_plus + S_minus = sum(nums)，可以解得 S_plus = (target + sum(nums)) / 2。

def fingTargetSumWays(nums:List[int], target:int)->int:
    total = sum(nums)
    if (target+total)%2 != 0 or total<target:
        return 0
    s_plus = (target+total)//2 
    dp = [0]*(s_plus + 1)
    dp[0] = 1
    for num in nums:
        for j in range(s_plus, num - 1, -1):
            dp[j] += dp[j-num]
    return dp[s_plus]

# leetcode 448 找到数组中消失的数字:
# 给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。
# 请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。
def findMissingNumbers(nums:List[int])->List[int]:
    n = len(nums)
    for num in nums:
        idx = abs(num) - 1
        if nums[idx]>0:
            nums[idx] = nums[idx]*-1
    missing_numbers = []
    for i in range(n):
        if nums[i]>0:
            missing_numbers.append(i+1)
    return missing_numbers

