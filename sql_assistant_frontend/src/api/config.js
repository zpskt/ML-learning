import axios from 'axios'
import API_CONFIG from '../config/api'

// 获取系统配置
export const getConfig = () => {
  return axios.get(API_CONFIG.CONFIG)
}

// 更新系统配置
export const updateConfig = (configData) => {
  return axios.put(API_CONFIG.CONFIG, configData)
}

export default {
  getConfig,
  updateConfig
}