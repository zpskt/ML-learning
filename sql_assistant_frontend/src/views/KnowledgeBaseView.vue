<template>
  <div class="knowledge-base-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="knowledge-base-card">
          <template #header>
            <div class="card-header">
              <span>知识库管理</span>
              <el-button type="primary" @click="showAddDialog" style="float: right; padding: 3px 0">添加条目</el-button>
            </div>
          </template>
          
          <el-table :data="knowledgeBaseEntries" style="width: 100%" v-loading="loading">
            <el-table-column prop="id" label="ID" width="60"></el-table-column>
            <el-table-column prop="question_template" label="问题模板" show-overflow-tooltip></el-table-column>
            <el-table-column prop="database" label="数据库" width="150"></el-table-column>
            <el-table-column prop="description" label="描述" show-overflow-tooltip></el-table-column>
            <el-table-column prop="created_at" label="创建时间" width="180"></el-table-column>
            <el-table-column label="操作" width="200">
              <template #default="scope">
                <el-button size="small" @click="showEditDialog(scope.row)">编辑</el-button>
                <el-button size="small" type="danger" @click="deleteEntry(scope.row.id)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 添加/编辑对话框 -->
    <el-dialog :title="dialogTitle" v-model="dialogVisible" width="600px">
      <el-form :model="currentEntry" label-width="100px">
        <el-form-item label="问题模板">
          <el-input v-model="currentEntry.question_template" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="SQL查询">
          <el-input v-model="currentEntry.sql_query" type="textarea" :rows="4"></el-input>
        </el-form-item>
        <el-form-item label="数据库">
          <el-select v-model="currentEntry.database" placeholder="请选择数据库">
            <el-option label="cloud_platform" value="cloud_platform"></el-option>
            <el-option label="storage" value="storage"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="currentEntry.description" type="textarea"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取 消</el-button>
          <el-button type="primary" @click="saveEntry">确 定</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'KnowledgeBaseView',
  data() {
    return {
      knowledgeBaseEntries: [],
      loading: false,
      dialogVisible: false,
      dialogTitle: '',
      currentEntry: {
        id: null,
        question_template: '',
        sql_query: '',
        database: 'cloud_platform',
        description: ''
      },
      isEditing: false
    }
  },
  mounted() {
    this.loadKnowledgeBase()
  },
  methods: {
    async loadKnowledgeBase() {
      this.loading = true
      try {
        const response = await axios.get('http://localhost:8000/knowledge')
        if (response.data.success) {
          this.knowledgeBaseEntries = response.data.data
        } else {
          this.$message.error('获取知识库失败: ' + response.data.error)
        }
      } catch (error) {
        console.error('获取知识库出错:', error)
        this.$message.error('获取知识库失败: ' + error.message)
      } finally {
        this.loading = false
      }
    },
    
    showAddDialog() {
      this.dialogTitle = '添加知识库条目'
      this.currentEntry = {
        id: null,
        question_template: '',
        sql_query: '',
        database: 'cloud_platform',
        description: ''
      }
      this.isEditing = false
      this.dialogVisible = true
    },
    
    showEditDialog(entry) {
      this.dialogTitle = '编辑知识库条目'
      this.currentEntry = { ...entry }
      this.isEditing = true
      this.dialogVisible = true
    },
    
    async saveEntry() {
      try {
        let response
        if (this.isEditing) {
          // 编辑条目
          response = await axios.put('http://localhost:8000/knowledge', {
            id: this.currentEntry.id,
            question_template: this.currentEntry.question_template,
            sql_query: this.currentEntry.sql_query,
            database: this.currentEntry.database,
            description: this.currentEntry.description
          })
        } else {
          // 添加新条目
          response = await axios.post('http://localhost:8000/knowledge', {
            question_template: this.currentEntry.question_template,
            sql_query: this.currentEntry.sql_query,
            database: this.currentEntry.database,
            description: this.currentEntry.description
          })
        }
        
        if (response.data.success) {
          this.$message.success(this.isEditing ? '更新成功' : '添加成功')
          this.dialogVisible = false
          this.loadKnowledgeBase() // 重新加载数据
        } else {
          this.$message.error((this.isEditing ? '更新失败' : '添加失败') + ': ' + response.data.error)
        }
      } catch (error) {
        console.error('保存条目出错:', error)
        this.$message.error((this.isEditing ? '更新' : '添加') + '失败: ' + error.message)
      }
    },
    
    async deleteEntry(id) {
      try {
        await this.$confirm('确认删除该条目吗？', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
        
        const response = await axios.delete('http://localhost:8000/knowledge', {
          data: { id: id }
        })
        
        if (response.data.success) {
          this.$message.success('删除成功')
          this.loadKnowledgeBase() // 重新加载数据
        } else {
          this.$message.error('删除失败: ' + response.data.error)
        }
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除条目出错:', error)
          this.$message.error('删除失败: ' + error.message)
        }
      }
    }
  }
}
</script>

<style scoped>
.knowledge-base-card {
  margin-bottom: 20px;
}

.card-header {
  font-weight: bold;
  font-size: 16px;
}
</style>