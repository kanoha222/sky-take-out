package com.sky.mapper;

import com.sky.entity.DishFlavor;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

/**
 * Description:
 * author kano
 * create 2024-08-25
 * project IntelliJ IDEA
 */
@Mapper
public interface DishFlavorMapper {
    void insertBatch(List<DishFlavor> dishFlavorList);
}
