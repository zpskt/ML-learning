<template>
  <div class="config-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="config-card">
          <template #header>
            <div class="card-header">
              <span>系统配置管理</span>
              <el-button type="primary" @click="saveConfig" style="float: right; padding: 3px 0" :loading="loading">保存配置</el-button>
            </div>
          </template>
          
          <el-tabs v-model="activeTab" @tab-click="handleTabClick">
            <!-- 数据库配置 -->
            <el-tab-pane label="数据库配置" name="database" ref="databaseTab">
              <el-alert
                title="注意：修改数据库配置后需要重启服务才能生效"
                type="warning"
                show-icon
                style="margin-bottom: 20px"
              />
              
              <div class="database-config-section">
                <h3>数据库连接配置</h3>
                <el-table :data="databaseList" style="width: 100%" border>
                  <el-table-column prop="name" label="数据库名称" width="200">
                    <template #default="scope">
                      <el-input v-model="scope.row.name" placeholder="请输入数据库名称"></el-input>
                    </template>
                  </el-table-column>
                  <el-table-column prop="uri" label="连接字符串">
                    <template #default="scope">
                      <el-input v-model="scope.row.uri" placeholder="请输入数据库连接字符串"></el-input>
                    </template>
                  </el-table-column>
                  <el-table-column label="操作" width="100">
                    <template #default="scope">
                      <el-button type="danger" @click="removeDatabase(scope.$index)" size="small">删除</el-button>
                    </template>
                  </el-table-column>
                </el-table>
                <div style="margin-top: 15px">
                  <el-button type="primary" @click="addDatabase">添加数据库</el-button>
                </div>
              </div>
            </el-tab-pane>
            
            <!-- 大模型配置 -->
            <el-tab-pane label="大模型配置" name="llm" ref="llmTab">
              <el-alert
                title="注意：修改大模型配置后立即生效"
                type="info"
                show-icon
                style="margin-bottom: 20px"
              />
              
              <div class="llm-config-section">
                <el-form :model="llmConfig" label-width="150px">
                  <el-form-item label="API Base URL">
                    <el-input v-model="llmConfig.base_url" placeholder="请输入API基础URL"></el-input>
                    <div class="form-item-tip">例如：http://localhost:11434/v1 或 https://api.deepseek.com</div>
                  </el-form-item>
                  
                  <el-form-item label="API Key">
                    <el-input v-model="llmConfig.api_key" placeholder="请输入API密钥" show-password></el-input>
                    <div class="form-item-tip">本地部署时可填写任意值</div>
                  </el-form-item>
                  
                  <el-form-item label="模型名称">
                    <el-input v-model="llmConfig.model" placeholder="请输入模型名称"></el-input>
                    <div class="form-item-tip">例如：deepseek-r1:7b 或 deepseek-chat</div>
                  </el-form-item>
                  
                  <el-form-item label="部署方式">
                    <el-radio-group v-model="llmConfig.is_local">
                      <el-radio :label="true">本地部署</el-radio>
                      <el-radio :label="false">云端部署</el-radio>
                    </el-radio-group>
                  </el-form-item>
                </el-form>
              </div>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { getConfig, updateConfig } from '../api/config'

export default {
  name: 'ConfigView',
  data() {
    return {
      activeTab: 'database',
      databaseConfigs: {},
      llmConfig: {
        base_url: '',
        api_key: '',
        model: '',
        is_local: true
      },
      loading: false
    }
  },
  computed: {
    databaseList() {
      return Object.keys(this.databaseConfigs).map(key => ({
        name: key,
        uri: this.databaseConfigs[key]
      }))
    }
  },
  mounted() {
    this.loadConfig()
  },
  methods: {
    async loadConfig() {
      this.loading = true
      try {
        const response = await getConfig()
        if (response.data.success) {
          const config = response.data.data
          this.databaseConfigs = { ...config.databases }
          this.llmConfig.base_url = config.base_url || ''
        } else {
          this.$message.error('获取配置失败: ' + response.data.error)
        }
      } catch (error) {
        console.error('获取配置出错:', error)
        this.$message.error('获取配置失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    addDatabase() {
      const newName = `new_database_${Object.keys(this.databaseConfigs).length + 1}`
      this.$set(this.databaseConfigs, newName, '')
    },
    
    removeDatabase(index) {
      const keys = Object.keys(this.databaseConfigs)
      const keyToRemove = keys[index]
      this.$delete(this.databaseConfigs, keyToRemove)
    },
    
    async saveConfig() {
      this.loading = true
      try {
        // 重新构建数据库配置对象
        const databases = {}
        this.databaseList.forEach(item => {
          databases[item.name] = item.uri
        })
        
        const configData = {
          databases: databases,
          base_url: this.llmConfig.base_url
        }
        
        const response = await updateConfig(configData)
        if (response.data.success) {
          this.$message.success('配置保存成功')
        } else {
          this.$message.error('保存配置失败: ' + response.data.error)
        }
      } catch (error) {
        console.error('保存配置出错:', error)
        this.$message.error('保存配置失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    // 处理标签页切换事件
    handleTabClick(tab) {
      // 使用nextTick确保DOM更新完成后再执行
      this.$nextTick(() => {
        // 延迟执行以避免ResizeObserver错误
        setTimeout(() => {
          // 如果切换到数据库配置标签页，重新计算表格布局
          if (tab.name === 'database' && this.$refs.databaseTab) {
            const table = this.$refs.databaseTab.$el.querySelector('.el-table')
            if (table && table.__vue__ && table.__vue__.doLayout) {
              table.__vue__.doLayout()
            }
          }
        }, 50)
      })
    }
  }
}
</script>

<style scoped>
.config-card {
  margin-bottom: 20px;
}

.card-header {
  font-weight: bold;
  font-size: 16px;
}

.database-config-section,
.llm-config-section {
  padding: 20px 0;
}

.form-item-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
</style>