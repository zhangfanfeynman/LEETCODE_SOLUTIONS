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

# leetcode 438 找到字符串中所有字母异位词
# 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

# 示例 1:

# 输入: s = "cbaebabacd", p = "abc"
# 输出: [0,6]
# 解释:
# 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
# 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
# 示例 2:

# 输入: s = "abab", p = "ab"
# 输出: [0,1,2]
# 解释:
# 起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
# 起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
# 起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。

from collections import defaultdict
def findAnagrams(s:str, p:str)->List[int]:
    p_count = defaultdict(int)
    s_count = defaultdict(int)
    for i in range(len(p)):
        p_count[p[i]] +=1
        s_count[s[i]] +=1
    result = []
    if s_count == p_count:
        result.append(0)
    for i in range(len(p), len(s)):
        s_count[s[i]] +=1
        s_count[s[i-len(p)]] -=1
        if s_count[s[i-len(p)]] == 0:
            del s_count[s[i-len(p)]]
        if s_count == p_count:
            result.append(i - len(p) +1)
    return result


# leetcode 437 路径总和II
# 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
# 前缀和概念：

# 前缀和是指从根节点到当前节点的路径上的所有节点值的和。

# 如果我们有前缀和 prefix_sum，那么对于当前节点，我们想知道是否存在一个之前的前缀和 prefix_sum - targetSum，这样从那个节点到当前节点的路径和就是 targetSum。

# 步骤：

# 使用一个哈希表 prefix_counts 来记录前缀和出现的次数。

# 递归遍历树，计算从根到当前节点的前缀和 current_sum。

# 检查 current_sum - targetSum 是否在 prefix_counts 中，如果在，说明存在路径满足条件。

# 更新 prefix_counts，递归左右子树，回溯时恢复 prefix_counts（因为路径必须是向下的，不能跨子树）。


def pathSum(root:TreeNode, targetsum:int)->int:
    prefix_counts = defaultdict(int)
    prefix_counts[0]=1
    def dfs(node, current_sum):
        if not node:
            return 0
        current_sum += node.val
        count = prefix_counts[targetsum - current_sum]
        prefix_counts[current_sum] +=1
        count += dfs(node.left, current_sum)
        count += dfs(node.right, current_sum)
        prefix_counts[current_sum] -=1
        if prefix_counts[current_sum] == 0:
            del prefix_counts[current_sum]
        return count
    return dfs(root, 0)

# leetcode 416. 分割等和子集
# 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
# 动态规划解法
# 子集和问题可以使用动态规划（Dynamic Programming, DP）来解决。具体思路如下：

# 计算总和：首先计算数组的总和 total。如果 total 是奇数，直接返回 false。否则，目标和 target = total / 2。

# 初始化DP数组：创建一个布尔型的二维数组 dp，其中 dp[i][j] 表示从前 i 个元素中选取一些，其和是否可以为 j。

# dp[0][0] = true：不选任何元素时，和为0。

# dp[0][j] = false（对于 j > 0）：没有元素可选时，无法得到正的和。

# 填充DP数组：

# 对于每个元素 nums[i-1]（因为 i 从1开始），和每个可能的和 j：

# 如果 j < nums[i-1]，则不能选择当前元素，dp[i][j] = dp[i-1][j]。

# 否则，可以选择不选或选当前元素：

# 不选：dp[i][j] = dp[i-1][j]

# 选：dp[i][j] = dp[i-1][j - nums[i-1]]

# 两者中有一个为 true 即可。

# 返回结果：dp[n][target] 就是答案，其中 n 是数组的长度。

def canPartition(nums:List[int])->bool:
    total = sum(nums)
    if total %2 !=0:
        return False
    target = total //2
    n = len(nums)
    dp = [[False]*(target+1) for _ in range(n+1)]
    dp[0][0] = True
    for i in range(1, n+1):
        for j in range(target+1):
            if j < nums[i-1]: # 不能选择当前元素
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j] or dp[i-1][j - nums[i-1]] # 不选或者选
    return dp[n][target]

# leetcode 406, 根据身高重建队列
# 假设有打乱顺序的一群人站成一个队列。每个人由一个整数对 (h, k) 表示，
# 其中 h 是这个人的身高，k 是排在这个人前面且身高大于或等于 h 的人数。
# 编写一个算法来重建这个队列。
def reconstructQueue(people:List[List[int]])->List[List[int]]:
    people.sort(key=lambda x: (-x[0], x[1])) # 按照身高降序排序，身高相同按k升序排序
    queue = []
    for person in people:
        queue.insert(person[1], person) # 将person插入到k位置
    return queue

# leetcode 394 字符串编码
# 给定一个经过编码的字符串，返回它解码后的字符串。

# 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
def decodeString(s:str)->str:
    stack = []
    current_num = 0
    current_str = ''
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ''
            current_num = 0
        elif char == ']':
            last_str, num = stack.pop()
            current_str = last_str + num*current_str
        else:
            current_str += char
    return current_str

# leetcode 347, 前k个高频元素
def topKFrequent(nums:List[int], k:int)->List[int]:
    freq_map = defaultdict(int)
    for num in nums:
        freq_map[num] +=1
    freq_buckets = [[] for _ in range(len(nums)+1)]
    for num, freq in freq_map.items():
        freq_buckets[freq].append(num)
    result = []
    for i in range(len(freq_buckets)-1, 0, -1):
        for num in freq_buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result
# leetcode 337，打家劫舍III
# 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

# 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 
# 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

# 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
# 可以考虑用后序遍历，左右根
def robTree(root: Optional[TreeNode]) -> int:
    def helper(node):
        if not node:
            return (0, 0)
        left = helper(node.left)
        right = helper(node.right)
        rob_current  = node.val + left[1]+right[1]
        not_rob_current = max(left) + max(right)
        return rob_current, not_rob_current
    return max(helper(root))

# leetcode 121. 买卖股票的最佳时机
# 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

# 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

# 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
def maxProfit(prices:List[int])->int:
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        if price < min_price: # 
            min_price = price
    return max_profit

# leetcode 309. 买卖股票的最佳时机 含冷冻期
# 给定一个整数数组prices，其中第 prices[i] 表示第 i 天的股票价格 。

# 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

# 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
# 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

# 这个问题是股票买卖问题的变种，加入了冷冻期的限制。我们需要在满足冷冻期条件下，通过多次买卖股票来获取最大利润。可以使用动态规划来解决，定义三种状态：

# 持有股票：当天结束时持有股票的最大利润。

# 不持有股票且处于冷冻期：当天卖出了股票，下一天不能买入。

# 不持有股票且不处于冷冻期：当天没有进行任何操作，可以自由买入。
def  maxProfitWithCooldown(prices:List[int])->int:
    """
    LeetCode 309: 含冷冻期的股票买卖的动态规划解法。

    状态定义（表示当天结束时的最大收益）：
    - hold: 持有股票的最大收益（当天结束）。
    - not_hold_cooldown: 不持股且当天处于冷冻期（刚卖出）的最大收益。
    - not_hold_no_cooldown: 不持股且当天不在冷冻期（可以买入）的最大收益。

    转移逻辑（处理第 i 天的价格 prices[i]）：
    - 持有状态可以保持（昨天就持有）或今天买入（昨天不持且无冷冻）：
        hold = max(hold, not_hold_no_cooldown - prices[i])
    - 不持且无冷冻可以来自昨天的不持且无冷冻（保持）或昨天处于冷冻期（冷却结束）：
        not_hold_no_cooldown = max(not_hold_no_cooldown, not_hold_cooldown)
    - 不持且处于冷冻期表示今天卖出，收益为昨天持有并在今天卖出：
        not_hold_cooldown = prev_hold + prices[i]

    最终答案是两种不持股状态的最大值（持有股票未变现不计）：
        return max(not_hold_no_cooldown, not_hold_cooldown)

    时间复杂度 O(n)，空间复杂度 O(1)。
    """
    if not prices:
        return 0
    n = len(prices)
    # 初始化（第0天结束时的三种状态）
    hold = -prices[0]                 # 第0天买入
    not_hold_cooldown = 0            # 第0天不可能处于冷冻期
    not_hold_no_cooldown = 0         # 第0天不持股且无冷冻

    for i in range(1, n):
        prev_hold = hold
        # 今天持有：要么延续昨天持有，要么昨天无冷冻且今天买入
        hold = max(hold, not_hold_no_cooldown - prices[i])
        # 今天不持且无冷冻：来自昨天不持（无冷冻）或昨天处于冷冻期（冷却结束）
        not_hold_no_cooldown = max(not_hold_no_cooldown, not_hold_cooldown)
        # 今天不持且处于冷冻期：今天卖出（昨天持有并今日卖出）
        not_hold_cooldown = prev_hold + prices[i]

    # 返回两种不持股状态的最大值
    return max(not_hold_no_cooldown, not_hold_cooldown)



# leetcode 300. 最长递增子序列

# 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

# 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

# 为了更好地理解，让我们看几个例子：

# 示例1：
# 输入：nums = [10,9,2,5,3,7,101,18]
# 输出：4
# 解释：最长递增子序列是 [2,3,7,101]，因此长度为4。
# 方法一：动态规划
# 动态规划是解决这类“最优化”问题的常用方法。我们需要找到一个状态表示和状态转移方程。

# 定义状态：

# 令 dp[i] 表示以 nums[i] 结尾的最长严格递增子序列的长度。

# 初始化：

# 对于每个 i，dp[i] 至少为1，因为至少可以包含 nums[i] 自己。
# 状态转移：

# 对于每个 i，我们需要检查所有 j < i：

# 如果 nums[j] < nums[i]，那么 nums[i] 可以接在 nums[j] 后面，形成一个新的递增子序列，长度为 dp[j] + 1。

# 我们需要在所有满足 nums[j] < nums[i] 的 j 中，选择最大的 dp[j] + 1 作为 dp[i] 的值。

def lengthOfLIS(nums:List[int])->int:
    if not nums:
        return 0
    n = len(nums)
    dp = [1]*n
    for i in range(1, n):
        for j in range(i):
            if nums[j]<nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

### leetcode 287, 寻找重复数
# 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
def findDuplicate(nums:List[int])->int:
    if not nums:
        return -1
    res = [False]*len(nums)
    for num in nums:
        if not res[num]:
            res[num] = True
        else:
            return num

# leetcode 283. 移动零
# # 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
def moveZeros(nums:List[int])->None:
    """
    Do not return anything, modify nums in-place instead.
    """
    if not nums:
        return 
    slow, fast = 0, 0
    for fast in range(len(nums)):
        if nums[fast]!= 0 :
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow +=1
    return

# leetcode 279. 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

# 这个问题可以看作是一个动态规划问题，因为：

# 示例演示
# 以 n = 12 为例：

# 初始化 dp = [0, ∞, ∞, ..., ∞]（长度为 n+1）。

# dp[0] = 0。

# 计算 dp[1] 到 dp[12]：

# i = 1：

# j = 1（1*1 <= 1）：

# dp[1] = min(∞, dp[0] + 1) = 1。

# dp[1] = 1。

# i = 2：

# j = 1：

# dp[2] = min(∞, dp[1] + 1) = 2。

# dp[2] = 2（2 = 1 + 1）。

# i = 3：

# j = 1：

# dp[3] = min(∞, dp[2] + 1) = 3。

# dp[3] = 3（3 = 1 + 1 + 1）。

# i = 4：

# j = 1：

# dp[4] = min(∞, dp[3] + 1) = 4。

# j = 2：

# dp[4] = min(4, dp[0] + 1) = 1。

# dp[4] = 1（4 = 4）。

# 继续计算直到 i = 12：

def numSquares(n:int)->int:
    dp = [float('inf')]*(n+1)
    dp[0] = 0
    for i in range(1, n+1):
        j = 1
        while j*j<=i:
            dp[i] = min(dp[i], dp[i - j*j]+1)
            j +=1
    return dp[n]

# leetcode 240. 搜索二维矩阵II
# 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
# 每行的元素从左到右升序排列。
# 每列的元素从上到下升序排列。
def searchMatrix(matrix:List[List[int]], target:int)->bool:
    if not matrix or not matrix[0]:
        return False
    rows,cols = len(matrix), len(matrix[0])
    row, col = 0, cols-1
    while row<rows and col>=0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            row +=1
        else:
            col -=1
    return False    

# leetcode 239 滑动窗口最大值
# 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

# 返回 滑动窗口中的最大值 。
def maxSlidingWindow(nums:List[int], k:int)->List[int]:
    if not nums:
        return []
    result = []
    window = []
    for i , num in enumerate(nums):
        while window and window[0] <= i-k:
            window.pop(0)
        while window and nums[window[-1]] < num:
            window.pop()
        window.append(i)
        if i >= k-1:
            result.append(nums[window[0]])
    return result

# leetcode 22. 括号生成
# 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

def generateParenthesis(n:int)->List[str]:
    result = []
    def backtrack(current, open_count, close_count):
        if len(current) == 2*n:
            result.append("".join(current))
            return
        if open_count < n:
            current.append('(')
            backtrack(current, open_count+1, close_count)
            current.pop()
        if close_count < n:
            current.append(')')
            backtrack(current, open_count, close_count+1)
            current.pop()
    backtrack([], 0, 0)
    return result
# leetcode 46, 全排列
# 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
def permute(nums:List[int])->List[List[int]]:
    result = []
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i]+remaining[i+1:])
            current.pop()
    backtrack([], nums)
    return result

# leetcode 42, 接雨水：
# 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
def trap(height:List[int])->int:
    if not height:
        return 0
    left, right  = 0, len(height)-1
    left_max, right_max = height[left], height[right]
    water_trapped = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water_trapped += left_max - height[left]
        else:
            right -=1
            right_max = max(right_max, height[right])
            water_trapped += right_max - height[right]
    return water_trapped


# leetcode 39. 组合总和
# 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 
# target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
# candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。

def combinationSum(candidates:List[int], target:int):
    def backtrack(remaining, combo, start):
        if remaining == 0:
            result.append(combo[:])
            return
        elif remaining < 0 :
            return
        for i in range(start, len(candidates)):
            combo.append(candidates[i])
            backtrack(remaining - candidates[i], combo, i)
            combo.pop()
    result = []
    backtrack(target, [], 0)
    return result

# leetcode 543. 二叉树的直径
# 给定一棵二叉树，你需要计算它的直径长度。
# 一棵二叉树的直径长度是任意两个结点路径长度中的最长值。这条路径可能穿过也可能不穿过根结点。
def diameterOfBinaryTree(root:Optional[TreeNode])->int:
    diameter = 0
    def depth(node):
        if not node:
            return 0
        nonlocal diameter
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        diameter = max(diameter, left_depth + right_depth)
        return max(left_depth, right_depth) + 1
    depth(root)
    return diameter

# leetcode 34, 在排序数组中查找元素的第一个和最后一个位置
# 给你一个按照 非递减顺序 排列的整数数组 nums，和一个目标值 target 。请你找出给定目标值在数组中的开始位置和结束位置。

def searchRange(nums:List[int], target:int)->List[int]:
    def findFirst():
        left, right = 0, len(nums) -1
        first_pos = -1
        while left <= right:
            mid = (left + right)//2
            if nums[mid] == target:
                first_pos = mid
                right = mid-1
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1
        return first_pos
    def findLast():
        left, right = 0, len(nums) -1
        last_pos = -1
        while left<= right:
            mid = (left + right)//2
            if nums[mid] == target:
                last_pos = mid
                left = mid+1
            elif nums[mid]>target:
                right = mid-1
            else:
                left = mid+1
        return last_pos
    return [findFirst(), findLast()]

