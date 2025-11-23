import axios from 'axios'
import API_CONFIG from '../config/api'

// 获取查询历史
export const getHistory = (params) => {
  return axios.get(API_CONFIG.HISTORY, { params })
}

// 获取查询历史（POST方式）
export const getHistoryPost = (historyData) => {
  return axios.post(API_CONFIG.HISTORY, historyData)
}

export default {
  getHistory,
  getHistoryPost
}