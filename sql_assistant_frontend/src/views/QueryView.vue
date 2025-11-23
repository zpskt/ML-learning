<template>
  <div class="query-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="query-card">
          <template #header>
            <div class="card-header">
              <span>数据库查询</span>
            </div>
          </template>
          
          <el-form :model="queryForm" label-width="120px">
            <el-form-item label="选择数据库">
              <el-select v-model="queryForm.dbName" placeholder="请选择数据库">
                <el-option label="Cloud Platform" value="cloud_platform"></el-option>
                <el-option label="Storage" value="storage"></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="查询问题">
              <el-input
                v-model="queryForm.question"
                type="textarea"
                placeholder="请输入您想查询的内容，例如：查询用户总数"
                :rows="4"
              ></el-input>
            </el-form-item>
            
            <el-form-item>
              <el-button type="primary" @click="submitQuery" :loading="loading">提交查询</el-button>
              <el-button @click="resetForm">重置</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
      
      <!-- 查询结果展示 -->
      <el-col :span="24" v-if="queryResult">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <span>查询结果</span>
            </div>
          </template>
          
          <div class="result-content">
            <el-tabs v-model="activeTab">
              <el-tab-pane label="自然语言回答" name="natural">
                <div class="natural-response">
                  {{ queryResult.data.natural_response }}
                </div>
              </el-tab-pane>
              
              <el-tab-pane label="SQL语句" name="sql">
                <div class="sql-response">
                  <el-input
                    v-model="queryResult.data.query"
                    type="textarea"
                    :rows="6"
                    readonly
                  ></el-input>
                  <div class="sql-actions">
                    <el-button @click="copySQL">复制SQL</el-button>
                    <el-button type="primary" @click="editSQL">修改SQL</el-button>
                  </div>
                </div>
              </el-tab-pane>
              
              <el-tab-pane label="查询结果" name="result">
                <div class="result-table">
                  <pre>{{ queryResult.data.result }}</pre>
                </div>
              </el-tab-pane>
              
              <el-tab-pane label="数据可视化" name="chart" v-if="showChartTab">
                <div class="chart-section">
                  <el-form :model="chartForm" label-width="120px" inline>
                    <el-form-item label="图表类型">
                      <el-select v-model="chartForm.type" placeholder="选择图表类型">
                        <el-option label="柱状图" value="bar"></el-option>
                        <el-option label="折线图" value="line"></el-option>
                        <el-option label="饼图" value="pie"></el-option>
                        <el-option label="散点图" value="scatter"></el-option>
                        <el-option label="直方图" value="histogram"></el-option>
                      </el-select>
                    </el-form-item>
                    
                    <el-form-item label="图表标题">
                      <el-input v-model="chartForm.title" placeholder="请输入图表标题"></el-input>
                    </el-form-item>
                    
                    <el-form-item>
                      <el-button type="primary" @click="generateChart">生成图表</el-button>
                    </el-form-item>
                  </el-form>
                  
                  <div class="chart-container" v-if="chartImage">
                    <img :src="'data:image/png;base64,' + chartImage" alt="查询结果图表" />
                  </div>
                </div>
              </el-tab-pane>
            </el-tabs>
          </div>
        </el-card>
      </el-col>
      
      <!-- 自定义SQL编辑 -->
      <el-col :span="24" v-if="showCustomSQLEditor">
        <el-card class="custom-sql-card">
          <template #header>
            <div class="card-header">
              <span>自定义SQL</span>
              <el-button @click="cancelCustomSQL" style="float: right; padding: 3px 0" type="text">取消</el-button>
            </div>
          </template>
          
          <el-input
            v-model="customSQL"
            type="textarea"
            :rows="8"
            placeholder="请输入自定义SQL语句"
          ></el-input>
          
          <div style="margin-top: 15px;">
            <el-button type="primary" @click="executeCustomSQL">执行SQL</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'QueryView',
  data() {
    return {
      queryForm: {
        dbName: 'cloud_platform',
        question: ''
      },
      loading: false,
      queryResult: null,
      activeTab: 'natural',
      showCustomSQLEditor: false,
      customSQL: '',
      chartForm: {
        type: 'bar',
        title: ''
      },
      chartImage: '',
      showChartTab: false
    }
  },
  methods: {
    async submitQuery() {
      if (!this.queryForm.question) {
        this.$message.warning('请输入查询问题')
        return
      }
      
      this.loading = true
      try {
        const response = await axios.post('http://localhost:8000/query', {
          user_question: this.queryForm.question,
          db_name: this.queryForm.dbName
        })
        
        this.queryResult = response.data
        this.showChartTab = true
        this.$message.success('查询成功')
      } catch (error) {
        this.$message.error('查询失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    resetForm() {
      this.queryForm = {
        dbName: 'cloud_platform',
        question: ''
      }
      this.queryResult = null
      this.showCustomSQLEditor = false
      this.customSQL = ''
      this.chartImage = ''
    },
    
    copySQL() {
      navigator.clipboard.writeText(this.queryResult.data.query)
      this.$message.success('SQL已复制到剪贴板')
    },
    
    editSQL() {
      this.customSQL = this.queryResult.data.query
      this.showCustomSQLEditor = true
    },
    
    cancelCustomSQL() {
      this.showCustomSQLEditor = false
      this.customSQL = ''
    },
    
    async executeCustomSQL() {
      if (!this.customSQL) {
        this.$message.warning('请输入SQL语句')
        return
      }
      
      this.loading = true
      try {
        const response = await axios.post('http://localhost:8000/execute-sql', {
          sql_query: this.customSQL,
          db_name: this.queryForm.dbName
        })
        
        this.queryResult = response.data
        this.showCustomSQLEditor = false
        this.$message.success('SQL执行成功')
      } catch (error) {
        this.$message.error('SQL执行失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    async generateChart() {
      if (!this.queryResult || !this.queryResult.data.result) {
        this.$message.warning('没有可可视化的数据')
        return
      }
      
      try {
        const response = await axios.post('http://localhost:8000/chart/generate', {
          query_result: this.queryResult.data.result,
          chart_type: this.chartForm.type,
          title: this.chartForm.title
        })
        
        if (response.data.success) {
          this.chartImage = response.data.data.image
          this.activeTab = 'chart'
          this.$message.success('图表生成成功')
        } else {
          this.$message.error('图表生成失败: ' + response.data.error)
        }
      } catch (error) {
        this.$message.error('图表生成失败: ' + error.message)
      }
    }
  }
}
</script>

<style scoped>
.query-card, .result-card, .custom-sql-card {
  margin-bottom: 20px;
}

.card-header {
  font-weight: bold;
  font-size: 16px;
}

.natural-response {
  font-size: 16px;
  line-height: 1.6;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.sql-actions {
  margin-top: 15px;
}

.chart-container {
  text-align: center;
  margin-top: 20px;
}

.chart-container img {
  max-width: 100%;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}
</style>