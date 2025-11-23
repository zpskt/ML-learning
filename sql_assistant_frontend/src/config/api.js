// API配置文件
const API_CONFIG = {
  // 开发环境
  development: {
    baseURL: 'http://localhost:8000'
  },
  // 生产环境
  production: {
    baseURL: 'http://localhost:8000'
  }
}

// 根据环境获取对应配置
const env = process.env.NODE_ENV || 'development'
export const API_BASE_URL = API_CONFIG[env].baseURL

export default {
  // 查询相关接口
  QUERY: `${API_BASE_URL}/query`,
  
  // 历史记录相关接口
  HISTORY: `${API_BASE_URL}/history`,
  
  // 知识库相关接口
  KNOWLEDGE: `${API_BASE_URL}/knowledge`,
  
  // 图表相关接口
  CHART_GENERATE: `${API_BASE_URL}/chart/generate`,
  
  // 自定义SQL执行接口
  EXECUTE_SQL: `${API_BASE_URL}/execute-sql`,
  
  // 配置管理接口
  CONFIG: `${API_BASE_URL}/config`,
  
  // 导出接口
  EXPORT_CSV: `${API_BASE_URL}/export/csv`
}