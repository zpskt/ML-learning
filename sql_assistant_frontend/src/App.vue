<template>
  <div id="app">
    <el-container>
      <el-header>
        <div class="header-content">
          <h1>SQL助手</h1>
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

.footer-content {
  text-align: center;
  color: #909399;
}
</style>