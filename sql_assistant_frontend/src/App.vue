<template>
  <div id="app">
    <el-container>
      <el-header>
        <div class="header-content">
          <h1>SQL助手</h1>
          <div class="nav-links">
            <el-button type="text" @click="$router.push('/')">首页</el-button>
            <el-button type="text" @click="$router.push('/query')">查询</el-button>
            <el-button type="text" @click="$router.push('/history')">历史</el-button>
            <el-button type="text" @click="$router.push('/knowledge')">知识库</el-button>
          </div>
        </div>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
      <el-footer>
        <div class="footer-content">
          <p>SQL助手 &copy; 2025</p>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-content h1 {
  color: #409eff;
  margin: 0;
}

.nav-links .el-button {
  margin-left: 20px;
  font-size: 16px;
  color: #606266;
}

.nav-links .el-button:hover {
  color: #409eff;
}

.footer-content {
  text-align: center;
  color: #909399;
}
</style>