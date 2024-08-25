package com.sky.service;

import com.sky.dto.DishDTO;

/**
 * Description:
 * author kano
 * create 2024-08-25
 * project IntelliJ IDEA
 */
public interface DishService {
    /**
     * 新增菜拼合对应的口味
     * @param dto
     */
    public void saveWithFlavor(DishDTO dto);
}
