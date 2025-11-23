import axios from 'axios'
import API_CONFIG from '../config/api'

// 查询数据库
export const queryDatabase = (queryData) => {
  return axios.post(API_CONFIG.QUERY, queryData)
}

// 执行自定义SQL
export const executeCustomSQL = (sqlData) => {
  return axios.post(API_CONFIG.EXECUTE_SQL, sqlData)
}

// 生成图表
export const generateChart = (chartData) => {
  return axios.post(API_CONFIG.CHART_GENERATE, chartData)
}

// 导出CSV
export const exportToCSV = (exportData) => {
  return axios.post(API_CONFIG.EXPORT_CSV, exportData, {
    responseType: 'blob'
  })
}

export default {
  queryDatabase,
  executeCustomSQL,
  generateChart,
  exportToCSV
}