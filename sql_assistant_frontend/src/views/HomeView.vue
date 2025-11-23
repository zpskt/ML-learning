<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <div class="welcome-card">
          <h2>欢迎使用SQL助手</h2>
          <p>这是一个智能的数据库查询工具，可以帮助您：</p>
          <ul>
            <li>使用自然语言查询数据库</li>
            <li>生成并执行SQL语句</li>
            <li>将查询结果可视化展示</li>
            <li>查看和管理查询历史</li>
            <li>管理知识库条目</li>
          </ul>
          <el-button type="primary" @click="goToQuery">开始查询</el-button>
          <el-button @click="goToHistory">查看历史</el-button>
          <el-button @click="goToKnowledgeBase">知识库管理</el-button>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: 'HomeView',
  methods: {
    goToQuery() {
      this.$router.push('/query')
    },
    goToHistory() {
      this.$router.push('/history')
    },
    goToKnowledgeBase() {
      this.$router.push('/knowledge')
    }
  }
}
</script>

<style scoped>
.welcome-card {
  background: #ffffff;
  border-radius: 4px;
  padding: 24px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  text-align: center;
}

.welcome-card h2 {
  color: #409eff;
  margin-bottom: 20px;
}

.welcome-card ul {
  text-align: left;
  max-width: 400px;
  margin: 20px auto;
}

.welcome-card li {
  margin-bottom: 10px;
}
</style>