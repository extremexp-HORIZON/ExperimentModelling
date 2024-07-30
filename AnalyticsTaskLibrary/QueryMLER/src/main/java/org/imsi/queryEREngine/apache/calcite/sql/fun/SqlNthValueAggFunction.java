/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.imsi.queryEREngine.apache.calcite.sql.fun;

import org.imsi.queryEREngine.apache.calcite.sql.SqlAggFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.util.Optionality;

/**
 * <code>NTH_VALUE</code> windowed aggregate function
 * returns the value of an expression evaluated at the {@code n}th row of the
 * window frame.
 */
public class SqlNthValueAggFunction extends SqlAggFunction {
	public SqlNthValueAggFunction(SqlKind kind) {
		super(kind.name(), null, kind, ReturnTypes.ARG0_NULLABLE_IF_EMPTY,
				null, OperandTypes.ANY_NUMERIC, SqlFunctionCategory.NUMERIC, false,
				true, Optionality.FORBIDDEN);
	}

	@Override public boolean allowsNullTreatment() {
		return true;
	}
}