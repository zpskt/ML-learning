<template>
  <div class="history-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="history-card">
          <template #header>
            <div class="card-header">
              <span>查询历史</span>
              <el-button @click="loadHistory" style="float: right; padding: 3px 0" type="text">刷新</el-button>
            </div>
          </template>
          
          <el-table :data="history" style="width: 100%" v-loading="loading">
            <el-table-column prop="timestamp" label="时间" width="180">
              <template #default="scope">
                {{ formatTime(scope.row.timestamp) }}
              </template>
            </el-table-column>
            <el-table-column prop="question" label="查询问题" show-overflow-tooltip></el-table-column>
            <el-table-column prop="database" label="数据库" width="120"></el-table-column>
            <el-table-column prop="duration" label="耗时(秒)" width="100"></el-table-column>
            <el-table-column label="操作" width="200">
              <template #default="scope">
                <el-button size="small" @click="viewDetail(scope.row)">查看详情</el-button>
                <el-button 
                  size="small" 
                  type="primary" 
                  @click="reExecute(scope.row)"
                  :loading="scope.row.id === executingId"
                >
                  重新执行
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          
          <div style="margin-top: 20px; text-align: center;">
            <el-pagination
              @current-change="handlePageChange"
              :current-page="currentPage"
              :page-size="pageSize"
              layout="total, prev, pager, next"
              :total="total"
            >
            </el-pagination>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 历史记录详情对话框 -->
    <el-dialog v-model="dialogVisible" title="查询详情" width="60%">
      <div v-if="selectedHistory">
        <el-tabs v-model="detailActiveTab">
          <el-tab-pane label="基本信息" name="basic">
            <el-descriptions :column="1" border>
              <el-descriptions-item label="时间">{{ formatTime(selectedHistory.timestamp) }}</el-descriptions-item>
              <el-descriptions-item label="查询问题">{{ selectedHistory.question }}</el-descriptions-item>
              <el-descriptions-item label="数据库">{{ selectedHistory.database }}</el-descriptions-item>
              <el-descriptions-item label="耗时">{{ selectedHistory.duration }} 秒</el-descriptions-item>
              <el-descriptions-item label="是否成功">
                <el-tag :type="selectedHistory.success ? 'success' : 'danger'">
                  {{ selectedHistory.success ? '成功' : '失败' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="错误信息" v-if="!selectedHistory.success">
                {{ selectedHistory.error }}
              </el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>
          
          <el-tab-pane label="SQL语句" name="sql">
            <el-input
              v-model="selectedHistory.query"
              type="textarea"
              :rows="8"
              readonly
            ></el-input>
          </el-tab-pane>
          
          <el-tab-pane label="查询结果" name="result">
            <pre>{{ selectedHistory.result }}</pre>
          </el-tab-pane>
        </el-tabs>
      </div>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { getHistory, getHistoryPost } from '../api/history'
import axios from 'axios'

export default {
  name: 'HistoryView',
  data() {
    return {
      history: [],
      loading: false,
      currentPage: 1,
      pageSize: 10,
      total: 0,
      dialogVisible: false,
      selectedHistory: null,
      detailActiveTab: 'basic',
      executingId: null
    }
  },
  mounted() {
    this.loadHistory()
  },
  methods: {
    async loadHistory() {
      this.loading = true
      try {
        const response = await getHistory({ limit: this.pageSize })
        if (response.data.success) {
          this.history = response.data.data
          this.total = response.data.data.length
        } else {
          this.$message.error('获取历史记录失败: ' + response.data.error)
        }
      } catch (error) {
        console.error('获取历史记录出错:', error)
        this.$message.error('获取历史记录失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    async loadHistoryByPost() {
      this.loading = true
      try {
        const response = await getHistoryPost({ limit: this.pageSize })
        if (response.data.success) {
          this.history = response.data.data
          this.total = response.data.data.length
        } else {
          this.$message.error('获取历史记录失败: ' + response.data.error)
        }
      } catch (error) {
        console.error('获取历史记录出错:', error)
        this.$message.error('获取历史记录失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleString('zh-CN')
    },
    
    handlePageChange(page) {
      this.currentPage = page
      // 这里可以实现分页逻辑
      this.loadHistory()
    },
    
    viewDetail(row) {
      this.selectedHistory = row
      this.dialogVisible = true
    },
    
    async reExecute(row) {
      this.executingId = row.id
      try {
        // 如果是自定义SQL查询
        if (row.custom_sql) {
          const response = await axios.post('http://localhost:8000/execute-sql', {
            sql_query: row.query,
            db_name: row.database
          })
          
          if (response.data.success) {
            this.$message.success('重新执行成功')
          } else {
            this.$message.error('重新执行失败: ' + response.data.error)
          }
        } else {
          // 如果是自然语言查询
          const response = await axios.post('http://localhost:8000/query', {
            user_question: row.question,
            db_name: row.database
          })
          
          if (response.data.success) {
            this.$message.success('重新执行成功')
          } else {
            this.$message.error('重新执行失败: ' + response.data.error)
          }
        }
      } catch (error) {
        this.$message.error('重新执行失败: ' + error.message)
      } finally {
        this.executingId = null
      }
    }
  }
}
</script>

<style scoped>
.history-card {
  margin-bottom: 20px;
}

.card-header {
  font-weight: bold;
  font-size: 16px;
}

pre {
  background-color: #f5f7fa;
  padding: 15px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 300px;
  overflow-y: auto;
}
</style>