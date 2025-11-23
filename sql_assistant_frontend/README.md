# SQL助手前端

这是一个基于Vue.js的前端项目，用于与SQL助手后端进行交互。

## 项目结构

```
sql_assistant_frontend/
├── public/                 # 静态资源文件
├── src/                    # 源代码目录
│   ├── assets/             # 静态资源
│   ├── components/         # 组件目录
│   ├── views/              # 页面视图
│   ├── router/             # 路由配置
│   ├── store/              # 状态管理
│   ├── App.vue             # 根组件
│   └── main.js             # 入口文件
├── package.json            # 项目配置文件
└── README.md               # 项目说明文档
```

## 功能特性

1. 自然语言数据库查询
2. SQL语句生成与执行
3. 查询结果可视化
4. 查询历史记录管理
5. 自定义SQL编辑与执行

## 安装依赖

```bash
npm install
```

## 启动开发服务器

```bash
npm run serve
```

## 构建生产版本

```bash
npm run build
```

## 技术栈

- Vue.js 3
- Element Plus UI组件库
- Axios HTTP客户端
- Vue Router 路由管理

## 与后端交互

前端通过以下API端点与后端进行交互：

- `POST /query` - 自然语言查询
- `POST /execute-sql` - 执行自定义SQL
- `GET /history` - 获取查询历史
- `POST /chart/generate` - 生成图表

## 注意事项

1. 确保后端服务运行在 `http://localhost:8000`
2. 浏览器需要支持JavaScript
3. 需要网络连接以加载外部资源