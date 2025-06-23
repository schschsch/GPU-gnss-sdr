# GNSS-SDR with GPU Acceleration

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **重要声明**  
> 本项目基于 [GNSS-SDR](https://github.com/gnss-sdr/gnss-sdr) 构建，
> 在 GNU General Public License v3.0 许可下发布。
>
> GPU 加速跟踪模块由 schschsch开发，同样遵循 GPLv3 许可。

基于GNSS-SDR，增加了veml_tracking_gpu模块，通过cuda将主要信号tracking计算通过GPU进行加速，实际效果在单channel情况下不佳，多channel情况下才能得到较好的效果。